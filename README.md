# ⛵ Smart Data Pipeline
![Tests](https://img.shields.io/badge/tests-102%20passed-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![HITL](https://img.shields.io/badge/HITL-%E2%9C%93-brightgreen)
![LLM](https://img.shields.io/badge/LLM-Gemini-orange)
![Prefect](https://img.shields.io/badge/orchestrator-Prefect-1f6feb)
![Python](https://img.shields.io/badge/python-3.12.6-blue)

> End-to-end ML pipeline for domain-driven text classification. Change one line in
> `config.yaml` and get a full annotated dataset, trained model, and interactive report
> for any topic.

## Quick Start
```bash
git clone https://github.com/AnastasiaButus/Smart-data-pipeline.git
cd smart-data-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# добавить GEMINI_API_KEY в .env

# Запустить весь пайплайн одной командой:
python pipeline/run_pipeline.py

# Или запустить UI:
streamlit run ui/app.py
```

## Data Sources

| Source | Type | License/Status |
|--------|------|----------------|
| HuggingFace (`dair-ai/emotion`) | dataset | Apache 2.0 |
| HuggingFace (`mteb/tweet_sentiment_extraction`) | dataset | MIT |
| StackExchange Sailing | API | CC BY-SA 4.0 |
| RSS feeds (`Yachting World`, `Cruising World`, `Sail Magazine`, `48 North`) | RSS scraping | editorial / educational use |
| Sailing Forums | web scraping | robots.txt checked, educational use |

## 🕷️ Scraping approach

The pipeline uses ethical scraping practices:
- `robots.txt` checked before every scrape
- Rate limiting: `time.sleep(1)` between requests
- User-Agent identifies the bot honestly
- Educational/non-commercial use only

**Supported scraping methods:**
- HuggingFace datasets API (official, no limits)
- StackExchange API (official, 300 req/day free)
- RSS feeds (`feedparser` — standard protocol)
- Forum scraping (`BeautifulSoup4` + `requests`)

**Hidden API pattern (stub in `scrape()`):**
Some sites expose internal REST APIs via browser Network tab. Pattern:
1. Open DevTools → Network tab
2. Find XHR requests with JSON responses
3. Copy as cURL → convert to Python `requests`
4. Add pagination support

This pattern is implemented as a stub in
`agents/data_collection_agent.py -> scrape()`
with a TODO comment for extension.

## Data Card

| Параметр | Значение |
|----------|----------|
| Домен | Sailing & yacht navigation |
| Язык | English |
| Источников | 8 (HF, StackExchange, RSS, форумы) |
| Строк собрано | 761 |
| Строк после чистки | 609 |
| Тематических строк | 220 (23.4%) |
| Классов | 5 + other_or_offtopic |
| Модель | sklearn TF-IDF + LogReg |
| Accuracy | 0.50 |
| F1 macro | 0.43 |
| HITL точек | 1 (`review_queue.csv`) |
| Тестов | 102 |

## Classes

| Класс | Описание | Строк |
|-------|----------|-------|
| navigation | Навигация, маршруты, карты | 51 |
| safety | Безопасность, спасение | 77 |
| equipment | Оборудование, паруса | 59 |
| weather | Погода, ветер, море | 24 |
| licensing | Лицензии, обучение | 9 |
| other_or_offtopic | Нетематические тексты | 389 |

## Architecture

The project is organized around 4 agents: data collection, data quality, annotation, and active learning.
Prefect orchestrates the end-to-end pipeline and provides one-command execution.
Gemini is used as a separate LLM layer for domain reformulation, EDA hypotheses, and compact advisory tasks.
Streamlit is the UI layer for HITL review, analytics, reporting, and chat with project context.

## ✨ Features

### 🤖 Multi-agent pipeline
- **DataCollectionAgent** — сбор из нескольких источников: HuggingFace datasets, StackExchange API, RSS-ленты и форумы
- **DataQualityAgent** — автоматическая чистка: HTML-артефакты, дубликаты, fuzzy matching и фильтрация коротких текстов
- **AnnotationAgent** — zero-shot авторазметка (`facebook/bart-large-mnli`) + confidence scoring + review queue
- **ActiveLearningAgent** — стратегии `entropy`, `margin`, `random` + learning curve и сравнение стратегий
- **ModelWrapper** — sklearn harness для `fit/predict/evaluate/explain/save/load`

### 🔍 Smart deduplication
- Exact deduplication by normalized text
- **Fuzzy matching** (`rapidfuzz`) — finds near-duplicates with >90% similarity, not just exact matches
- Catches variants like `Sailing in bad weather` vs `Sailing in bad weather!` that exact match misses

### 🧠 LLM-powered (Gemini)
- Переформулировка темы пользователя в ML-постановку с классами и ключевыми словами
- Объяснение проблем качества данных и рекомендация стратегии чистки
- Генерация EDA-гипотез на русском языке
- Чат с данными прямо в дашборде
- Fallback chain: `gemini-2.5-flash -> gemini-flash-latest -> gemma`

### 👤 Human-in-the-Loop (HITL)
- Примеры с `confidence < threshold` попадают в `review_queue.csv`
- Streamlit-интерфейс позволяет принять или исправить метку
- Фильтрация по классу и источнику
- Сохранение правок и скачивание CSV для ручной проверки

### 📊 Interactive EDA Report
- 8 интерактивных Plotly-графиков
- WordCloud с фильтрацией HTML-мусора
- LLM-гипотезы на русском языке
- Сворачиваемые секции и export в standalone HTML

### 📋 Report Builder
- Пользователь выбирает секции галочками
- Форматы: HTML / Markdown / Telegram
- Таблица источников с лицензиями и статусом скрапинга
- Data Card с полными метриками проекта

### ⚙️ Orchestration
- Prefect flow запускает весь pipeline одной командой
- `skip_hitl` и `skip_al` позволяют пропускать тяжёлые шаги
- `ContextMemory` сохраняет результаты каждого этапа
- Graceful degradation не даёт пайплайну падать из-за внешних зависимостей

## Pipeline Steps

- [x] Step 2.1: DataQualityAgent — HTML cleanup, dedup, **fuzzy matching**, filtering

## Reports

| Report | Description |
|--------|-------------|
| `reports/domain_reformulation.md` | Детали domain reformulation |
| `reports/quality_report.md` | Сводка до/после чистки данных |
| `reports/annotation_spec.md` | Спецификация разметки |
| `reports/eda_report.html` | Интерактивный EDA-отчёт |
| `reports/model_metrics.json` | Финальные метрики модели |
| `models/classifier.pkl` | Обученная sklearn-модель |

## UI — Streamlit Dashboard
```bash
streamlit run ui/app.py
```

Дашборд включает 4 вкладки:
- `🚀 Онбординг` — задать тему и получить рекомендации по источникам
- `🔍 Проверка меток` — HITL-очередь с правкой и сохранением меток
- `📊 Аналитика` — графики, лицензии, конструктор отчёта
- `💬 Чат с LLM` — вопросы о данных, гипотезах и качестве

## Retrospective

### Что сработало хорошо
- Fallback chain Gemini — пайплайн не падает из-за LLM недоступности
- ContextMemory — каждый агент знает, что сделал предыдущий
- Graceful degradation везде — тесты проходят без обязательных реальных API-вызовов
- Streamlit UI — ключевые действия доступны без CLI

### Что можно улучшить
- Тематических данных мало (23.4%) — нужно больше яхтинг-специфичных источников
- `navigation` и `weather` классы слабые (`F1 < 0.35`) из-за малого числа примеров
- Gemini нестабилен и часто уходит в fallback — стоит рассмотреть более стабильный тариф или провайдера
- `sailingforums` сильно проседает после чистки (`20 -> 5` строк)

### Что бы сделал иначе
- Начал бы с более тематических HuggingFace датасетов вместо `emotion/tweets`
- Добавил бы augmentation для малых классов
- Реализовал бы fine-tune DistilBERT как следующий baseline upgrade
