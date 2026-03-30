"""DataCollectionAgent — collects raw text from HuggingFace, forum scraping, and RSS feeds."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from loguru import logger


class DataCollectionAgent:
    """Collects raw unlabeled text from HuggingFace datasets, forum scraping, and RSS feeds."""

    _COLUMNS = ["id", "text", "label", "source", "collected_at"]
    _USER_AGENT = "smart-data-pipeline/0.1 (educational project)"

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load configuration and initialise logger. No network connections opened here."""
        self._config_path = Path(config_path)
        with open(self._config_path) as fh:
            self._cfg = yaml.safe_load(fh)
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
                dataset = load_dataset(name, split=split, trust_remote_code=True)
                df = dataset.to_pandas()
                if text_col not in df.columns:
                    # Try to find a suitable text column
                    text_col = next(
                        (c for c in df.columns if df[c].dtype == object), None
                    )
                    if text_col is None:
                        logger.warning("No text column found in {}", name)
                        continue
                    logger.warning(
                        "Column 'text' not found in {}, using '{}'", name, text_col
                    )
                df = df[[text_col]].rename(columns={text_col: "text"})
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
    #  Source: Cruisers Forum scraping                                     #
    # ------------------------------------------------------------------ #

    def scrape_forum(self) -> pd.DataFrame:
        """Scrape thread titles and first posts from Cruisers Forum sections.

        Respects robots.txt. Returns empty DataFrame if crawling is disallowed
        or any network error occurs.
        """
        scraping_cfg = self._cfg["sources"]["scraping"]
        if not scraping_cfg.get("enabled", True):
            logger.info("Scraping source disabled in config")
            return self._empty_df()

        forum_cfg = scraping_cfg.get("cruisers_forum", {})
        if not forum_cfg.get("enabled", True):
            logger.info("Cruisers Forum scraping disabled in config")
            return self._empty_df()

        base_url: str = forum_cfg.get("base_url", "https://www.cruisersforum.com")
        sections: list[str] = forum_cfg.get("sections", ["/forums/f19/", "/forums/f4/"])
        pages_per_section: int = forum_cfg.get("pages_per_section", 3)

        # --- robots.txt check ---
        if not self._is_crawl_allowed(base_url, self._USER_AGENT):
            logger.warning(
                "robots.txt disallows crawling {}, skipping forum scrape", base_url
            )
            return self._empty_df()

        session = requests.Session()
        session.headers.update({"User-Agent": self._USER_AGENT})
        records: list[dict] = []

        for section in sections:
            for page in range(1, pages_per_section + 1):
                if page > 1:
                    url = f"{base_url}{section}index{page}.html"
                else:
                    url = f"{base_url}{section}"
                try:
                    logger.info("Scraping forum page: {}", url)
                    resp = session.get(url, timeout=10)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Try specific Cruisers Forum selectors first, then fallback
                    thread_links = soup.select("a.title, a[id^='thread_title_']")
                    if not thread_links:
                        thread_links = [
                            a for a in soup.find_all("a", href=True)
                            if "/forums/" in a.get("href", "") and "showthread" in a.get("href", "")
                        ]
                    for link in thread_links[:10]:
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
                    continue
                except Exception as exc:
                    logger.warning("Unexpected error scraping {}: {}", url, exc)
                    continue

        if not records:
            logger.info(
                "No records collected from Cruisers Forum (possibly blocked or empty)"
            )
            return self._empty_df()

        df = pd.DataFrame(records)
        logger.info("Cruisers Forum: {} rows collected", len(df))
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
                logger.info("RSS {}: {} entries collected", domain, len(parsed.entries))
            except Exception as exc:
                logger.warning("Failed to fetch RSS {}: {}", feed_url, exc)

        if not records:
            return self._empty_df()

        df = pd.DataFrame(records)
        logger.info("RSS total: {} rows collected", len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Generic scrape (extensibility stub)                                 #
    # ------------------------------------------------------------------ #

    def scrape(self, url: str, selector: str = "") -> pd.DataFrame:
        """Universal scraping entry point — dispatches to known scrapers by URL.

        # TODO: extend for other sites in step 1.2
        """
        logger.info("generic scrape called for url={}", url)
        if "cruisersforum" in url.lower():
            return self.scrape_forum()
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
        source_fns = {
            "huggingface": self.fetch_huggingface,
            "forum": self.scrape_forum,
            "rss": self.fetch_rss,
        }
        results: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
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
            "step": "1.1",
            "status": "done",
            "metrics": {
                "total_rows": len(df),
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
        """Generate synthetic sailing-domain texts as fallback when all sources fail."""
        import random

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
                "robots.txt at {} disallows crawling for agent '{}'", robots_url, user_agent
            )
        return allowed
