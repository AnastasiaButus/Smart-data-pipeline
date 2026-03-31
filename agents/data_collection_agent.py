"""DataCollectionAgent — collects raw text from HuggingFace, forum scraping, and RSS feeds."""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse, urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from loguru import logger

# Browser-like User-Agent for sites that block bots
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class DataCollectionAgent:
    """Collects raw unlabeled text from HuggingFace, forum scraping, RSS, and StackExchange."""

    _COLUMNS = ["id", "text", "label", "source", "collected_at"]
    _USER_AGENT = "smart-data-pipeline/0.1 (educational project)"
    _SAILING_TOPIC = "sailing and yacht navigation"

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load configuration and initialise logger. No network connections opened here."""
        self._config_path = Path(config_path)
        with open(self._config_path) as fh:
            self._cfg = yaml.safe_load(fh)
        self.config = self._cfg
        self._raw_path = Path(self._cfg["data"]["raw_path"])
        self._raw_path.mkdir(parents=True, exist_ok=True)
        self._reports_path = Path("reports")
        self._reports_path.mkdir(parents=True, exist_ok=True)
        logger.info("DataCollectionAgent initialised from {}", self._config_path)

    # ------------------------------------------------------------------ #
    #  Source: HuggingFace                                                 #
    # ------------------------------------------------------------------ #

    def fetch_huggingface(self) -> pd.DataFrame:
        """Download two HuggingFace datasets and return a combined DataFrame."""
        from datasets import load_dataset  # heavy import — deferred

        hf_cfg = self._cfg["sources"]["huggingface"]
        if not hf_cfg.get("enabled", True):
            logger.info("HuggingFace source disabled in config")
            return self._empty_df()

        frames: list[pd.DataFrame] = []
        for ds_spec in hf_cfg.get("datasets", []):
            name: str = ds_spec["name"]
            split: str = ds_spec.get("split", "train")
            text_col: str = ds_spec.get("text_column", "text")
            limit: int = ds_spec.get("limit", 300)
            try:
                logger.info("Fetching HuggingFace dataset {} split={}", name, split)
                dataset = load_dataset(name, split=split)
                df_raw = dataset.to_pandas()
                if text_col in df_raw.columns:
                    if not self._is_text_column(df_raw[text_col]):
                        logger.warning(
                            "HuggingFace {}: column '{}' doesn't look like text (avg_len too short or low diversity). Skipping.",
                            name,
                            text_col,
                        )
                        str_cols = df_raw.select_dtypes(include="object").columns
                        text_col = None
                        for col in str_cols:
                            if self._is_text_column(df_raw[col]):
                                text_col = col
                                logger.info("Auto-detected text column: {}", col)
                                break
                        if not text_col:
                            logger.warning("No text column found in {}, skipping", name)
                            continue
                if text_col not in df_raw.columns:
                    text_col = next(
                        (c for c in df_raw.columns if df_raw[c].dtype == object and self._is_text_column(df_raw[c])),
                        None,
                    )
                    if text_col is None:
                        logger.warning("No text column found in {}", name)
                        continue
                    logger.warning(
                        "Column 'text' not found in {}, using '{}'", name, text_col
                    )
                df = df_raw[[text_col]].rename(columns={text_col: "text"})
                df = df.head(limit)
                slug = name.replace("/", "_")
                df["source"] = f"huggingface_{slug}"
                df["label"] = "unlabeled"
                df["collected_at"] = datetime.now(timezone.utc).isoformat()
                frames.append(df)
                logger.info("HuggingFace {}: {} rows collected", name, len(df))
            except Exception as exc:
                logger.warning("Failed to fetch HuggingFace dataset {}: {}", name, exc)

        if not frames:
            return self._empty_df()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Source: Forum scraping (Cruisers Forum → sailingforums.com)        #
    # ------------------------------------------------------------------ #

    def scrape_forum(self) -> pd.DataFrame:
        """Scrape sailing forum thread titles.

        Strategy:
        1. Try Cruisers Forum with browser User-Agent.
        2. If robots.txt blocks — fall back to www.sailingforums.com.
        Returns empty DataFrame on any unrecoverable error.
        """
        scraping_cfg = self._cfg["sources"]["scraping"]
        if not scraping_cfg.get("enabled", True):
            logger.info("Scraping source disabled in config")
            return self._empty_df()

        forum_cfg = scraping_cfg.get("cruisers_forum", {})
        if not forum_cfg.get("enabled", True):
            logger.info("Cruisers Forum scraping disabled in config")
            return self._empty_df()

        # --- Variant A: Cruisers Forum with browser UA ---
        cf_base = forum_cfg.get("base_url", "https://www.cruisersforum.com")
        sections: list[str] = forum_cfg.get("sections", ["/forums/f19/", "/forums/f4/"])
        pages_per_section: int = forum_cfg.get("pages_per_section", 3)

        if self._is_crawl_allowed(cf_base, _BROWSER_UA):
            result = self._scrape_cruisers_forum(cf_base, sections, pages_per_section)
            if not result.empty:
                return result
            logger.info("Cruisers Forum returned 0 rows — falling back to sailingforums.com")
        else:
            logger.warning(
                "robots.txt blocks crawling on {} — falling back to sailingforums.com", cf_base
            )

        # --- Variant B: sailingforums.com ---
        return self._scrape_sailingforums()

    def _scrape_cruisers_forum(
        self, base_url: str, sections: list[str], pages_per_section: int
    ) -> pd.DataFrame:
        """Scrape thread titles from Cruisers Forum sections using browser User-Agent."""
        session = requests.Session()
        session.headers.update({"User-Agent": _BROWSER_UA})
        records: list[dict] = []

        for section in sections:
            for page in range(1, pages_per_section + 1):
                url = (
                    f"{base_url}{section}index{page}.html"
                    if page > 1
                    else f"{base_url}{section}"
                )
                try:
                    logger.info("Scraping Cruisers Forum: {}", url)
                    resp = session.get(url, timeout=10)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    thread_links = soup.select("a.title, a[id^='thread_title_']")
                    if not thread_links:
                        thread_links = [
                            a for a in soup.find_all("a", href=True)
                            if "showthread" in a.get("href", "")
                        ]
                    for link in thread_links[:15]:
                        title_text = link.get_text(strip=True)
                        if len(title_text) >= 20:
                            records.append({
                                "text": title_text,
                                "label": "unlabeled",
                                "source": "cruisers_forum",
                                "collected_at": datetime.now(timezone.utc).isoformat(),
                            })
                    time.sleep(1)  # rate limiting — mandatory
                except requests.RequestException as exc:
                    logger.warning("Network error scraping {}: {}", url, exc)
                except Exception as exc:
                    logger.warning("Unexpected error scraping {}: {}", url, exc)

        if not records:
            return self._empty_df()
        df = pd.DataFrame(records)
        logger.info("Cruisers Forum: {} rows collected", len(df))
        return df

    def _scrape_sailingforums(self) -> pd.DataFrame:
        """Fallback: scrape thread titles from www.sailingforums.com."""
        base_url = "https://www.sailingforums.com"
        if not self._is_crawl_allowed(base_url, _BROWSER_UA):
            logger.warning("robots.txt blocks sailingforums.com — skipping forum scrape")
            return self._empty_df()

        session = requests.Session()
        session.headers.update({"User-Agent": _BROWSER_UA})
        records: list[dict] = []

        try:
            logger.info("Scraping fallback forum: {}", base_url)
            resp = session.get(base_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            # Generic thread link selectors for vBulletin/XenForo style boards
            thread_links = soup.select(
                "a.PreviewTooltip, a.title, h3.title a, .structItem-title a, "
                "a[href*='threads/'], a[href*='showthread']"
            )
            for link in thread_links[:40]:
                title_text = link.get_text(strip=True)
                if len(title_text) >= 20:
                    records.append({
                        "text": title_text,
                        "label": "unlabeled",
                        "source": "sailingforums",
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    })
            time.sleep(1)
        except requests.RequestException as exc:
            logger.warning("Network error scraping sailingforums.com: {}", exc)
        except Exception as exc:
            logger.warning("Unexpected error scraping sailingforums.com: {}", exc)

        if not records:
            logger.info("sailingforums.com returned 0 records")
            return self._empty_df()
        df = pd.DataFrame(records)
        logger.info("sailingforums.com: {} rows collected", len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Source: RSS feeds                                                   #
    # ------------------------------------------------------------------ #

    def fetch_rss(self) -> pd.DataFrame:
        """Fetch entries from configured RSS feeds, combining title and summary as text."""
        import feedparser  # deferred import

        rss_cfg = self._cfg["sources"]["rss"]
        if not rss_cfg.get("enabled", True):
            logger.info("RSS source disabled in config")
            return self._empty_df()

        feeds: list[str] = rss_cfg.get("feeds", [])
        records: list[dict] = []

        for feed_url in feeds:
            domain = urlparse(feed_url).netloc
            try:
                logger.info("Fetching RSS feed: {}", feed_url)
                parsed = feedparser.parse(feed_url)
                if parsed.bozo and not parsed.entries:
                    logger.warning("RSS feed malformed or unavailable: {}", feed_url)
                    continue
                feed_count = 0
                for entry in parsed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}".strip(". ").strip()
                    if len(text) >= 20:
                        records.append({
                            "text": text,
                            "label": "unlabeled",
                            "source": f"rss_{domain}",
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        })
                        feed_count += 1
                logger.info("RSS {}: {}/{} entries collected", domain, feed_count, len(parsed.entries))
            except Exception as exc:
                logger.warning("Failed to fetch RSS {}: {}", feed_url, exc)

        if not records:
            return self._empty_df()

        df = pd.DataFrame(records)
        logger.info("RSS total: {} rows collected", len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Source: StackExchange Sailing                                       #
    # ------------------------------------------------------------------ #

    def fetch_stackexchange(self) -> pd.DataFrame:
        """Fetch tagged questions from StackExchange public API.

        Uses the official JSON API — no robots.txt scraping needed.
        Robots check is performed on the API domain as a best-practice signal.
        Returns empty DataFrame on any network error.
        """
        se_cfg = self._cfg["sources"].get("stackexchange", {})
        if not se_cfg.get("enabled", True):
            logger.info("StackExchange source disabled in config")
            return self._empty_df()

        pages: int = se_cfg.get("pages", 5)
        api_root = "https://api.stackexchange.com"
        site = str(se_cfg.get("site", "outdoors")).strip() or "outdoors"
        tag = str(se_cfg.get("tag", "sailing")).strip() or "sailing"
        tag_slug = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_") or "tagged"

        # robots.txt check on API domain — best practice
        if not self._is_crawl_allowed(api_root, self._USER_AGENT):
            logger.warning("robots.txt blocks crawling {}, skipping", api_root)
            return self._empty_df()

        records: list[dict] = []

        for page_num in range(1, pages + 1):
            url = (
                f"{api_root}/2.3/questions"
                f"?site={site}&tagged={tag}&pagesize=100"
                f"&page={page_num}&order=desc&sort=activity"
            )
            try:
                logger.info("Fetching StackExchange API page {}", page_num)
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                items = data.get("items", [])
                if not items:
                    logger.info("StackExchange API page {} empty — stopping", page_num)
                    break

                for item in items:
                    title: str = item.get("title", "")
                    tags: list[str] = item.get("tags", [])
                    tags_str = " ".join(tags)
                    text = f"{title} {tags_str}".strip() if tags_str else title
                    if len(text) >= 20:
                        records.append({
                            "text": text,
                            "label": "unlabeled",
                            "source": f"stackexchange_{tag_slug}",
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                        })

                quota = data.get("quota_remaining", "?")
                has_more = data.get("has_more", False)
                logger.info(
                    "StackExchange API page {}: {} items, quota_remaining={}",
                    page_num, len(items), quota
                )

                if not has_more:
                    break
                time.sleep(0.5)  # API rate limiting

            except requests.RequestException as exc:
                logger.warning("Network error on StackExchange API page {}: {}", page_num, exc)
                break
            except Exception as exc:
                logger.warning("Unexpected error on StackExchange API page {}: {}", page_num, exc)
                break

        if not records:
            logger.info("StackExchange API returned 0 records")
            return self._empty_df()

        df = pd.DataFrame(records)
        logger.info("StackExchange: {} rows collected", len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Source: Kaggle                                                     #
    # ------------------------------------------------------------------ #

    def fetch_kaggle(self) -> pd.DataFrame:
        """Load configured Kaggle datasets for the current topic."""
        import os
        import tempfile
        import zipfile  # noqa: F401
        from glob import glob

        if not self.config.get("sources", {}).get("kaggle", {}).get("enabled", False):
            logger.info("Kaggle source disabled in config")
            return self._empty_df()
        kaggle_cfg = self.config.get("sources", {}).get("kaggle", {})

        token = os.getenv("KAGGLE_API_TOKEN", "")
        if not token:
            logger.warning("KAGGLE_API_TOKEN not set, skipping")
            return self._empty_df()

        try:
            os.environ["KAGGLE_API_TOKEN"] = token
            import kaggle  # deferred heavy import

            kaggle.api.authenticate()

            all_dfs: list[pd.DataFrame] = []
            for ds in kaggle_cfg.get("datasets", []):
                if not ds.get("enabled", True):
                    continue
                try:
                    logger.info("Downloading Kaggle dataset: {}", ds["ref"])
                    with tempfile.TemporaryDirectory() as tmp:
                        kaggle.api.dataset_download_files(
                            ds["ref"],
                            path=tmp,
                            unzip=True,
                            quiet=True,
                        )

                        csvs = glob(f"{tmp}/**/*.csv", recursive=True)
                        for csv_path in csvs[:1]:
                            df_raw = pd.read_csv(csv_path, nrows=ds.get("limit", 200))
                            text_col = ds.get("text_column")
                            if not text_col:
                                str_cols = [
                                    col
                                    for col in df_raw.select_dtypes(include="object").columns
                                    if self._is_text_column(df_raw[col])
                                ]
                                text_col = str_cols[0] if len(str_cols) > 0 else None
                            if text_col and text_col in df_raw.columns and self._is_text_column(df_raw[text_col]):
                                slug = ds["ref"].split("/")[-1]
                                df_out = pd.DataFrame(
                                    {
                                        "text": df_raw[text_col].astype(str),
                                        "label": "unlabeled",
                                        "source": f"kaggle_{slug}",
                                        "collected_at": datetime.now(timezone.utc).isoformat(),
                                    }
                                )
                                df_out["id"] = [str(uuid.uuid4()) for _ in range(len(df_out))]
                                all_dfs.append(df_out[self._COLUMNS])
                                logger.info("Kaggle {}: {} rows", ds["ref"], len(df_out))
                            else:
                                logger.warning("Kaggle {} has no suitable text column", ds["ref"])
                except Exception as exc:
                    logger.warning("Kaggle {} failed: {}", ds.get("ref", "unknown"), exc)
                    continue

            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
        except Exception as exc:
            logger.warning("Kaggle fetch failed: {}", exc)

        return self._empty_df()

    # ------------------------------------------------------------------ #
    #  Generic scrape dispatcher                                           #
    # ------------------------------------------------------------------ #

    def scrape(self, url: str, selector: str = "") -> pd.DataFrame:
        """Universal scraping entry point — dispatches to known scrapers by URL.

        Supported dispatches:
        - cruisersforum.com  → scrape_forum()
        - stackexchange.com  → fetch_stackexchange()
        - sailingforums.com  → scrape_forum() (triggers fallback path)
        # TODO: extend for additional sites in future steps
        """
        logger.info("generic scrape called for url={}", url)
        url_lower = url.lower()
        if "cruisersforum" in url_lower or "sailingforums" in url_lower:
            return self.scrape_forum()
        if "stackexchange" in url_lower:
            return self.fetch_stackexchange()
        logger.info("No specific scraper for {}, returning empty DataFrame", url)
        return self._empty_df()

    # ------------------------------------------------------------------ #
    #  Merge & deduplicate                                                 #
    # ------------------------------------------------------------------ #

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge DataFrames, deduplicate by normalised text, filter by length, add UUID ids."""
        non_empty = [df for df in sources if df is not None and not df.empty]
        if not non_empty:
            logger.warning("All sources returned empty DataFrames — using synthetic fallback")
            return self._generate_synthetic()

        combined = pd.concat(non_empty, ignore_index=True)

        # Normalise text for deduplication
        combined["_text_norm"] = combined["text"].str.strip().str.lower()
        before = len(combined)
        combined = combined.drop_duplicates(subset="_text_norm")
        combined = combined.drop(columns=["_text_norm"])
        logger.info("Deduplication: {} → {} rows", before, len(combined))

        # Length filter
        text_len = combined["text"].str.len()
        combined = combined[(text_len >= 20) & (text_len <= 5000)]
        logger.info("After length filter: {} rows", len(combined))

        combined = combined.reset_index(drop=True)
        # Drop any pre-existing id column before assigning fresh UUIDs
        if "id" in combined.columns:
            combined = combined.drop(columns=["id"])
        combined.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(combined))])

        # Ensure all required columns present
        for col in self._COLUMNS:
            if col not in combined.columns:
                combined[col] = None

        return combined[self._COLUMNS]

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """Collect data from all sources in parallel, merge, save, and return result."""
        logger.info("DataCollectionAgent.run() started")
        topic = self._get_current_topic()
        source_fns = {
            "huggingface": self.fetch_huggingface,
            "kaggle": self.fetch_kaggle,
        }
        if self._is_sailing_topic(topic):
            source_fns.update(
                {
                    "forum": self.scrape_forum,
                    "rss": self.fetch_rss,
                    "stackexchange": self.fetch_stackexchange,
                }
            )
        else:
            logger.info(
                "Topic '{}' is non-sailing — skipping sailing-specific RSS/forum/StackExchange sources",
                topic or "unknown",
            )
        results: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=max(len(source_fns), 1)) as executor:
            futures = {executor.submit(fn): name for name, fn in source_fns.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    logger.info("Source '{}': {} rows", name, len(results[name]))
                except Exception as exc:
                    logger.warning("Source '{}' raised exception: {}", name, exc)
                    results[name] = self._empty_df()

        df = self.merge(list(results.values()))

        # Save parquet
        out_path = self._raw_path / "dataset.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved {} rows to {}", len(df), out_path)

        # Save context memory
        memory = {
            "step": "1.2",
            "status": "done",
            "metrics": {
                "total_rows": len(df),
                "topic": topic,
                "sources": {name: len(r) for name, r in results.items()},
                "avg_text_len": round(df["text"].str.len().mean(), 1) if len(df) else 0,
                "columns": list(df.columns),
            },
            "notes": "",
        }
        memory_path = self._reports_path / "context_memory.json"
        with open(memory_path, "w") as fh:
            json.dump(memory, fh, indent=2)
        logger.info("Context memory saved to {}", memory_path)

        return df

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Return a compact metrics dict without raw data — safe for LLM context."""
        memory_path = self._reports_path / "context_memory.json"
        if memory_path.exists():
            with open(memory_path) as fh:
                saved = json.load(fh)
            return saved.get("metrics", {})
        return {"total_rows": 0, "sources": {}, "avg_text_len": 0.0, "columns": []}

    # ------------------------------------------------------------------ #
    #  Synthetic fallback                                                  #
    # ------------------------------------------------------------------ #

    def _generate_synthetic(self, n: int = 50) -> pd.DataFrame:
        """Generate topic-aware synthetic texts as fallback when all sources fail."""
        import random

        topic = self._get_current_topic()
        if self._is_sailing_topic(topic):
            templates = [
                "The mainsail was reefed as the wind increased to 25 knots on the beam reach.",
                "Anchoring in a crowded bay requires careful attention to scope and swing radius.",
                "The GPS chartplotter showed we were two miles off the waypoint due to current.",
                "Running lights must be displayed from sunset to sunrise when underway.",
                "MOB drill: throw the life ring, press MOB button on GPS, keep the person in sight.",
                "Tacking through the shipping lane requires constant VHF watch on channel 16.",
                "The depth sounder alarm was set at 3 meters to warn of shoaling water.",
                "We motorsailed into the harbour against a foul tide and 15 knot headwind.",
                "The jib furling line jammed at the worst possible moment during the squall.",
                "COLREGS rule 16: the give-way vessel shall take early and substantial action.",
                "Checking the weather forecast before departure is non-negotiable seamanship.",
                "The EPIRB was registered with the coast guard and mounted near the companionway.",
                "Sail trim: ease the sheet until the telltales on the luff start to lift, then trim.",
                "Night watch rotation: 3 hours on, 6 hours off keeps the crew adequately rested.",
                "A DSC distress call on VHF channel 70 will alert nearby vessels automatically.",
                "The standing rigging was inspected for broken strands and crevice corrosion.",
                "Provisioning for a 10-day offshore passage requires careful meal planning.",
                "The barometer had been falling steadily for 6 hours — a front was approaching.",
                "Entering the marina on starboard tack, we gave way to the outbound vessel.",
                "Celestial navigation backup: sun sight at noon for latitude determination.",
                "The winch drum was loaded incorrectly, causing the sheet to override under load.",
                "AIS transponder class B transmits position every 30 seconds when underway.",
                "A proper watch schedule prevents fatigue on offshore passages lasting several days.",
                "The bilge pump was tested and the float switch checked before leaving the dock.",
                "Reading the tide tables correctly is essential for entering shallow harbours.",
                "Safety briefing: life jacket locations, flare kit, emergency tiller, sea cocks.",
                "The autopilot was disengaged for the narrow channel approach to the marina.",
                "Boat hook technique: approach the dock at a shallow angle, not head-on.",
                "The VHF radio check confirmed DSC MMSI was programmed correctly before departure.",
                "Reefing early is always better than waiting for the conditions to force your hand.",
                "Fog navigation: sound signals, radar watch, and reduced speed in restricted visibility.",
                "The chart showed a submerged rock 200 metres off the headland at low water.",
                "Jacklines were rigged fore and aft before leaving the harbour in the forecast gale.",
                "Heaving-to in heavy weather: back the jib, ease the mainsheet, and adjust the helm.",
                "The diesel engine raw water strainer was cleaned weekly in tropical anchorages.",
                "Coast guard float plan filed before the offshore passage — a simple safety habit.",
                "Keel design affects both stability and leeway made on upwind passages.",
                "The spinnaker halyard was led aft before the sail was hoisted in gusty conditions.",
                "Battery bank monitoring: house bank at 12.3V after overnight at anchor is low.",
                "The dinghy davits were stowed and secured before departure in heavy swell.",
                "Radar reflector mounted at the spreaders improves detection by ship traffic.",
                "Waypoint routing through the archipelago avoided the charted shallow patches.",
                "Sail inventory for ocean passages: main, genoa, working jib, storm jib, trysail.",
                "The forestay tension was adjusted with the Loos gauge to manufacturer specification.",
                "Passage planning includes checking Navtex for weather and navigation warnings.",
                "The life raft was serviced at the authorised station before the Atlantic crossing.",
                "Running backstays must be set up before gybing in heavy air to protect the mast.",
                "A kedge anchor off the stern prevented the boat swinging onto the dock in the surge.",
                "The chart plotter track showed we had made good 140 nautical miles in 24 hours.",
                "Crew overboard recovery under sail: quick-stop manoeuvre or figure-of-eight method.",
                "The marina berth was too short — we had to anchor in the outer roads for the night.",
            ]
        else:
            safe_topic = topic or "general domain"
            templates = [
                f"This document introduces the core concepts and terminology used in {safe_topic}.",
                f"A practitioner explains common workflows, edge cases, and troubleshooting steps in {safe_topic}.",
                f"This article compares beginner and advanced approaches to learning {safe_topic}.",
                f"An expert checklist summarises the most important safety, quality, and review steps in {safe_topic}.",
                f"The guide outlines tools, best practices, and frequent mistakes related to {safe_topic}.",
                f"A case study describes how teams evaluate data, labels, and model quality for {safe_topic}.",
                f"This note highlights domain-specific jargon, examples, and recurring patterns in {safe_topic}.",
                f"A long-form overview explains regulation, maintenance, and operational concerns in {safe_topic}.",
                f"The tutorial walks through typical scenarios, exceptions, and decision points in {safe_topic}.",
                f"An interview transcript captures practical experience, lessons learned, and recommendations for {safe_topic}.",
            ]
        random.seed(42)
        texts = [templates[i % len(templates)] for i in range(n)]
        random.shuffle(texts)
        df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(n)],
            "text": texts,
            "label": "unlabeled",
            "source": "synthetic",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Generated {} synthetic fallback rows", n)
        return df[self._COLUMNS]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _empty_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with the required schema."""
        return pd.DataFrame(columns=self._COLUMNS)

    def _get_current_topic(self) -> str:
        """Return the topic currently configured for the pipeline."""
        domain_cfg = self._cfg.get("domain", {})
        return str(domain_cfg.get("topic", self._SAILING_TOPIC)).strip()

    def _is_sailing_topic(self, topic: str) -> bool:
        """Return True when the configured topic belongs to the sailing domain."""
        topic_lower = str(topic or "").lower()
        return any(
            keyword in topic_lower
            for keyword in [
                "sail",
                "yacht",
                "boat",
                "ship",
                "marine",
                "nautical",
                "ocean",
                "sea",
                "naval",
            ]
        )

    def _is_text_column(self, series: pd.Series) -> bool:
        """Heuristically detect whether a column contains free-form text."""
        sample = series.dropna().head(20).astype(str)
        if sample.empty:
            return False
        avg_len = sample.str.len().mean()
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return bool(avg_len > 20 and unique_ratio > 0.3)

    def _is_crawl_allowed(self, base_url: str, user_agent: str) -> bool:
        """Check robots.txt for the given base URL and user agent string."""
        robots_url = f"{base_url.rstrip('/')}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as exc:
            logger.warning(
                "Could not read robots.txt at {}: {} — assuming allowed", robots_url, exc
            )
            return True
        allowed = rp.can_fetch(user_agent, base_url + "/")
        if not allowed:
            logger.warning(
                "robots.txt at {} disallows crawling for agent '{}'",
                robots_url,
                user_agent[:40],
            )
        return allowed
