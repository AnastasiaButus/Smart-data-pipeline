"""Tests for project scaffold: folder structure, config integrity, env template."""

from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).parent.parent


def test_required_directories_exist():
    """All required project directories must be present."""
    required_dirs = [
        "agents",
        "core",
        "pipeline",
        "ui",
        "data/raw",
        "data/labeled",
        "models",
        "reports",
        "tests",
        "notebooks",
    ]
    for d in required_dirs:
        assert (ROOT / d).is_dir(), f"Missing directory: {d}"


def test_required_python_files_exist():
    """All required Python module files must be present."""
    required_files = [
        "agents/__init__.py",
        "agents/data_collection_agent.py",
        "agents/data_quality_agent.py",
        "agents/annotation_agent.py",
        "agents/al_agent.py",
        "core/__init__.py",
        "core/llm_client.py",
        "core/context_memory.py",
        "core/model_wrapper.py",
        "pipeline/__init__.py",
        "pipeline/run_pipeline.py",
        "ui/app.py",
    ]
    for f in required_files:
        assert (ROOT / f).is_file(), f"Missing file: {f}"


def test_config_yaml_exists_and_is_valid():
    """config.yaml must exist and parse without errors."""
    config_path = ROOT / "config.yaml"
    assert config_path.is_file(), "config.yaml not found"
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    assert config is not None, "config.yaml is empty"


def test_config_yaml_required_sections():
    """All required top-level sections must be present in config.yaml."""
    config_path = ROOT / "config.yaml"
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    required_sections = [
        "project",
        "domain",
        "sources",
        "data",
        "annotation",
        "active_learning",
        "model",
        "llm",
        "pipeline",
    ]
    for section in required_sections:
        assert section in config, f"Missing config section: {section}"


def test_config_domain_has_topic_and_classes():
    """domain section must have 'topic' and 'classes' keys."""
    config_path = ROOT / "config.yaml"
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    domain = config["domain"]
    assert "topic" in domain, "domain.topic missing"
    assert "classes" in domain, "domain.classes missing"
    assert isinstance(domain["classes"], list), "domain.classes must be a list"
    assert len(domain["classes"]) > 0, "domain.classes must not be empty"


def test_env_example_exists():
    """.env.example must exist."""
    assert (ROOT / ".env.example").is_file(), ".env.example not found"


def test_env_example_has_required_keys():
    """.env.example must contain all required environment variable names."""
    env_path = ROOT / ".env.example"
    content = env_path.read_text()
    required_keys = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "GEMINI_API_KEY",
    ]
    for key in required_keys:
        assert key in content, f"Missing key in .env.example: {key}"


def test_gitignore_exists_and_covers_sensitive_paths():
    """.gitignore must exist and protect .env and data directories."""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore not found"
    content = gitignore_path.read_text()
    assert ".env" in content, ".gitignore must include .env"
    assert "data/raw/*" in content, ".gitignore must ignore data/raw/*"
    assert "data/labeled/*" in content, ".gitignore must ignore data/labeled/*"


def test_requirements_txt_exists():
    """requirements.txt must exist and be non-empty."""
    req_path = ROOT / "requirements.txt"
    assert req_path.is_file(), "requirements.txt not found"
    assert req_path.stat().st_size > 0, "requirements.txt is empty"
