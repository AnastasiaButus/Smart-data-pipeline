# smart-data-pipeline

![Status](https://img.shields.io/badge/status-WIP-yellow)

An end-to-end configurable text classification pipeline with LLM-assisted annotation, active learning, and a human-in-the-loop Streamlit dashboard. The topic and label classes are fully configurable — set your domain in `config.yaml` and the pipeline handles data collection, quality checks, annotation, model training, and review.

## Setup

```bash
git clone https://github.com/your-username/smart-data-pipeline.git
cd smart-data-pipeline
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your API keys, then edit `config.yaml` to set your classification topic and classes.
