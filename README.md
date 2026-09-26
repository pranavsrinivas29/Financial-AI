# Financial AI — ML, NLP, RAG, Agentic AI & MLOps

An end-to-end Financial AI platform combining **Machine Learning, Natural Language Processing, Retrieval-Augmented Generation (RAG), Agentic AI, explainability, and MLOps**.

The project analyzes financial market data, company fundamentals, financial news, and SEC filings to produce point-in-time financial analysis while avoiding future-data leakage.

The system is designed as a **development and portfolio environment**. Core AI components are implemented and executable locally, while selected infrastructure components such as Kubernetes, Helm, Terraform, and Ansible are provided as development/reference templates rather than representing a production cloud deployment.

---

## 1. Project Overview

Traditional financial AI projects often focus on a single capability such as stock prediction, sentiment analysis, or document question answering.

This project combines multiple independent AI capabilities:

- Machine Learning for market direction prediction
- Machine Learning for volatility forecasting
- NLP-based financial news sentiment analysis
- RAG over SEC 10-Q and 8-K filings
- Agentic AI for orchestrating specialized AI components
- Explainable AI using SHAP
- Experiment tracking and model registry using MLflow
- Data and model monitoring using Evidently
- API and system monitoring using Prometheus and Grafana
- FastAPI backend
- Streamlit user interface
- Docker-based local deployment
- Kubernetes and Helm deployment templates
- CI/CD templates
- Terraform infrastructure templates
- Ansible configuration-management templates

A fundamental design requirement is **point-in-time correctness**.

The system ensures that analysis for a given date only uses information that would have been available at that point in time.

---

# 2. High-Level Architecture

```text
                         User
                           |
                           v
                    Streamlit UI
                           |
                           v
                     FastAPI API
                           |
                           v
                 LangGraph Supervisor
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      ML Agent         NLP Agent        RAG Agent
          |                |                |
          v                v                v
     XGBoost           FinBERT       SEC Retrieval
 Direction / Vol.     Sentiment      Qdrant + BM25
          |                                 |
          |                           Cross Encoder
          |                                 |
          +---------------+-----------------+
                          |
                          v
                  Analysis / Report
                          |
                          v
                     Streamlit
```

Supporting MLOps layer:

```text
DVC
 |
 +---- Data Versioning

MLflow
 |
 +---- Experiment Tracking
 +---- Hyperparameters
 +---- Metrics
 +---- Artifacts
 +---- Model Registry

Evidently
 |
 +---- Data Quality
 +---- Feature Drift
 +---- Prediction Drift
 +---- Model Monitoring

Prometheus
 |
 +---- API Metrics
 +---- Latency
 +---- Error Rates
 +---- Agent/RAG Metrics
 |
 v
Grafana
```

Deployment architecture:

```text
Application
    |
    v
Docker
    |
    v
Docker Compose
    |
    +-------------------+
    |                   |
Local Development   Kubernetes
                        |
                        v
                       Helm

Infrastructure templates
        |
        +---- Terraform
        |
        +---- Ansible
```

---

# 3. Core Design Principles

## Point-in-Time Correctness

The project avoids look-ahead bias and future-data leakage.

For an analysis date `T`:

```text
Market data     <= T
News            <= T
SEC filings     <= T
Fundamentals    available <= T
ML features     <= T
```

Future market data is used only for supervised-learning targets during historical model training.

For example, a 20-day target is:

```text
target_return_20d =
    price(t + 20) / price(t) - 1
```

but future prices are never included in model features.

---

## Separation of AI Responsibilities

The system intentionally keeps ML, NLP and RAG separate.

```text
ML
├── Market direction prediction
└── Volatility forecasting

NLP
└── Financial news sentiment

RAG
└── SEC filing evidence retrieval

Agentic AI
└── Orchestrates the above capabilities
```

News sentiment is **not fed into the ML model**.

This separation improves:

- modularity
- explainability
- independent evaluation
- maintainability
- leakage control
- agent-level orchestration

---

# 4. Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| ML | XGBoost, scikit-learn |
| NLP | FinBERT, Hugging Face |
| Embeddings | Sentence Transformers / BGE |
| RAG | Qdrant, BM25 |
| Reranking | CrossEncoder |
| LLM | Ollama / local LLMs |
| Agentic AI | LangGraph |
| Backend | FastAPI |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Monitoring | Evidently |
| Observability | Prometheus, Grafana |
| Data Versioning | DVC |
| Explainability | SHAP |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Packaging | Helm |
| CI/CD | GitHub Actions |
| Infrastructure as Code | Terraform |
| Configuration Management | Ansible |

---

# 5. Project Phases

## Phase 1 — Environment, Repository and Data Versioning

Initial development environment and repository structure were established.

Implemented:

- Python environment
- Git repository
- modular `src/` structure
- configuration management
- DVC
- market dataset versioning

Example:

```text
data/raw/market/AAPL.parquet
```

Large datasets are managed using DVC rather than committed directly to Git.

---

# 6. Phase 2 — Point-in-Time Data Engineering

The data layer provides historical information while respecting the requested analysis date.

Primary sources include:

- market data
- SPY market context
- SEC filings
- SEC Company Facts
- financial news

The temporal layer ensures:

```text
information_available_at <= as_of_date
```

## SEC Filings

The system supports:

- 10-Q
- 8-K

For every analysis date, the system retrieves only the latest eligible filing available at that point.

Example:

```text
Analysis date: 2026-07-15

10-Q filed 2026-05-01    -> allowed
10-Q filed 2026-07-31    -> rejected

8-K filed 2026-06-10     -> allowed
8-K filed 2026-07-20     -> rejected
```

This prevents future filing information from leaking into historical analysis.

---

# 7. Phase 3 — NLP Financial Sentiment

Financial news sentiment is analyzed independently using **FinBERT**.

Pipeline:

```text
Financial News
      |
      v
Point-in-Time Filter
      |
      v
Deduplication
      |
      v
FinBERT
      |
      v
Positive / Neutral / Negative
      |
      v
Aggregate Sentiment
```

News articles must satisfy:

```text
published_at <= as_of_date
```

Outputs include probabilities for:

- positive sentiment
- neutral sentiment
- negative sentiment

The NLP subsystem is independent of the predictive ML models.

---

# 8. Phase 4 — Predictive Machine Learning

Two ML problems are implemented.

## Direction Classification

Predict whether the future return over the forecasting horizon is positive or negative.

```text
Target:

1 -> positive future return
0 -> negative/non-positive future return
```

## Volatility Regression

Predict future realized annualized volatility.

The default forecasting horizon is:

```text
20 trading days
```

---

## ML Features

### Market Features

Examples:

```text
return_1d
return_5d
return_20d
momentum_20d
momentum_50d
volatility_20d
volatility_60d
ma_20_50_ratio
volume_change_5d
volume_ratio_20d
drawdown_1y
```

### Market Context

```text
spy_return_20d
spy_momentum_20d
spy_volatility_20d
```

### Fundamental Features

```text
revenue_growth
operating_margin
net_margin
current_ratio
liabilities_to_equity
cash_to_current_liabilities
```

Fundamental information is aligned using its actual availability date rather than blindly using the accounting period end date.

---

# 9. Time-Series Validation

Random train/test splitting is intentionally avoided.

The project uses:

```text
Walk-Forward Validation
+
Purged Temporal Splits
+
Final Untouched Holdout
```

For each validation period:

```text
Training
------------>

             Validation
             ---------->

                         Future
```

Training rows are purged if their target horizon overlaps the validation period.

For example:

```text
target_end_date < validation_start
```

This prevents subtle target leakage across temporal boundaries.

---

# 10. ML Models

Direction models include:

```text
DummyClassifier
LogisticRegression
XGBoost Classifier
```

Volatility models include:

```text
DummyRegressor
LinearRegression
XGBoost Regressor
```

Simple baselines are deliberately retained so that complex models can be compared against meaningful benchmarks.

---

# 11. Hyperparameter Optimization

Hyperparameters are selected using historical walk-forward validation.

They are not retuned for every prediction.

Lifecycle:

```text
Historical Data
      |
      v
Walk-Forward CV
      |
      v
Hyperparameter Search
      |
      v
Best Configuration
      |
      v
Final Holdout
      |
      v
Model Registry
```

Hyperparameter retuning can occur periodically or when monitoring indicates degradation.

---

# 12. MLflow

MLflow provides experiment and model lifecycle tracking.

Tracked information includes:

- model type
- ticker
- forecast horizon
- training period
- validation period
- hyperparameters
- evaluation metrics
- artifacts
- model versions

Registered models include:

```text
financial-direction-model
financial-volatility-model
```

MLflow is used for:

```text
Experiments
    ↓
Runs
    ↓
Metrics / Parameters
    ↓
Model Artifacts
    ↓
Model Registry
```

---

# 13. Model Retraining

A feature row becomes trainable only after its future target is known.

For a 20-trading-day horizon:

```text
Feature at T
       |
       +---- target available around T + 20 trading days
```

At retraining time, only matured labels are used.

The production-style design separates:

```text
Inference
```

from:

```text
Scheduled Retraining
```

rather than training a new model for every API request.

---

# 14. Phase 5 — SEC Financial RAG

The RAG system allows natural-language questions over SEC filings.

Architecture:

```text
Question
   |
   v
Point-in-Time Filing Selection
   |
   v
SEC 10-Q / 8-K
   |
   v
HTML Parsing
   |
   v
Section-Aware Chunking
   |
   +----------------+
   |                |
   v                v
Embeddings         BM25
   |                |
   v                |
Qdrant              |
   |                |
   +-------+--------+
           |
           v
 Reciprocal Rank Fusion
           |
           v
    CrossEncoder
      Reranking
           |
           v
        Top-K
           |
           v
      Local LLM
           |
           v
 Answer + Sources
```

---

## Section-Aware Chunking

SEC filings are divided around sections such as:

```text
Item 1
Item 1A
Item 2
Item 2.02
...
```

Chunks retain metadata including:

```text
ticker
filing_type
filed_date
accession_number
section
chunk_id
source
```

---

## Hybrid Retrieval

The system combines:

```text
Dense Vector Retrieval
+
BM25 Lexical Retrieval
```

using rank fusion.

This improves retrieval of both:

- semantically similar content
- exact financial terminology

---

## Reranking

Initial candidates are reranked using a CrossEncoder:

```text
Top 20 candidates
        |
        v
CrossEncoder
        |
        v
Top 5
        |
        v
LLM
```

This reduces irrelevant context sent to the language model.

---

## Grounded Generation

The LLM is instructed to:

- use only retrieved SEC context
- avoid unsupported financial claims
- return sources
- treat retrieved documents as data, not instructions
- avoid following instructions embedded inside retrieved text

This provides a basic defense against indirect prompt injection.

---

# 15. Phase 6 — Agentic AI

LangGraph orchestrates the independent AI capabilities.

Conceptually:

```text
                 Supervisor
                     |
          +----------+----------+
          |          |          |
          v          v          v
       ML Agent   NLP Agent   RAG Agent
          |          |          |
          +----------+----------+
                     |
                     v
               Final Analysis
```

## ML Agent

Provides:

- direction prediction
- prediction probability
- volatility forecast

## NLP Agent

Provides:

- financial news sentiment
- sentiment probabilities
- aggregate sentiment

## RAG Agent

Provides:

- SEC evidence
- relevant filing sections
- grounded answers
- source references

The supervisor determines which capabilities are required for a user request.

---

# 16. Phase 7 — FastAPI Backend

FastAPI exposes the AI capabilities through REST APIs.

Representative endpoints:

```text
GET  /health

POST /predict
POST /sentiment
POST /ask
POST /analyze

GET  /models
```

Example analysis request:

```json
{
    "ticker": "AAPL",
    "as_of_date": "2026-07-15",
    "query": "Analyze the company's current financial situation."
}
```

The API layer separates the AI implementation from the frontend.

---

# 17. Phase 8 — Streamlit Application

Streamlit provides an interactive interface for the complete system.

The dashboard integrates:

```text
Market Analysis
ML Predictions
News Sentiment
SEC Filing Q&A
Agentic Analysis
```

Typical flow:

```text
Select Ticker
      |
Select Analysis Date
      |
      v
Run Analysis
      |
      +---- ML Prediction
      |
      +---- NLP Sentiment
      |
      +---- SEC RAG
      |
      +---- Agent Report
```

This provides a single demonstration interface for the complete Financial AI system.

---

# 18. Phase 9 — Explainable AI

SHAP is used to explain the predictive ML models.

SHAP is applied to:

- direction model
- volatility model

It is **not** used as an explanation mechanism for RAG or the language model.

Example:

```text
Direction Prediction
        |
        v
     XGBoost
        |
        v
      SHAP
        |
        +---- momentum_20d
        +---- spy_return_20d
        +---- volatility_20d
        +---- revenue_growth
        +---- operating_margin
```

Possible visualizations include:

- global feature importance
- local waterfall plots
- feature contribution plots

---

# 19. Phase 10 — Monitoring and Observability

The project separates model monitoring from infrastructure monitoring.

## MLflow

Used for:

```text
Experiments
Model versions
Parameters
Metrics
Artifacts
Registry
```

## Evidently

Development/reference monitoring includes:

```text
Feature drift
Prediction drift
Missing features
Data quality
Prediction distributions
Actual vs predicted
Classification confidence
Volatility forecast error
```

## Prometheus

Application-level metrics can include:

```text
API request count
API latency
API errors
ML inference latency
RAG retrieval latency
LLM latency
Agent execution time
```

## Grafana

Prometheus metrics can be visualized through Grafana dashboards.

For this development project, monitoring configurations and scripts demonstrate the intended observability architecture without requiring a continuously running production environment.

---

# 20. Phase 11 — Security

Security is intentionally lightweight and appropriate for a development environment.

The architecture considers:

```text
JWT authentication
Role-based access control
Input validation
Rate limiting
Environment-based secrets
CORS configuration
Audit logging
Prompt-injection protection
RAG context isolation
LLM guardrails
```

## RAG Security Principle

Retrieved content is considered **untrusted data**.

```text
SEC / external document
        |
        v
      DATA

        X

  SYSTEM INSTRUCTION
```

Instructions contained in retrieved documents must never override system-level instructions.

---

# 21. Phase 12 — Docker

The application is designed to be containerized for reproducible local execution.

Potential services include:

```text
FastAPI
Streamlit
MLflow
Qdrant
Redis
Prometheus
Grafana
```

Docker Compose can be used to run the development stack.

Conceptually:

```text
docker compose up
        |
        +---- FastAPI
        +---- Streamlit
        +---- MLflow
        +---- Qdrant
        +---- Redis
        +---- Prometheus
        +---- Grafana
```

---

# 22. Phase 13 — Kubernetes and Helm

Kubernetes and Helm are included as deployment templates for demonstrating container orchestration and application packaging.

The project does **not claim a production Kubernetes deployment**.

Example Helm structure:

```text
helm/
└── financial-ai/
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── api-deployment.yaml
        ├── api-service.yaml
        ├── streamlit-deployment.yaml
        ├── streamlit-service.yaml
        ├── qdrant.yaml
        ├── redis.yaml
        └── ingress.yaml
```

Development validation can include:

```bash
helm lint helm/financial-ai
```

A local Kubernetes environment such as kind or Minikube can optionally be used.

---

# 23. Phase 14 — CI/CD

GitHub Actions provides the CI/CD reference workflow.

The development pipeline can include:

```text
Git Push
    |
    v
Lint
    |
    v
Unit Tests
    |
    v
ML Tests
    |
    v
Docker Build
    |
    v
Security Scan
    |
    v
Helm Lint
    |
    v
Terraform Validation
    |
    v
Ansible Syntax Check
```

Cloud deployment is intentionally optional.

---

# 24. Phase 15 — Terraform

Terraform demonstrates Infrastructure as Code.

Terraform is maintained as an **infrastructure template**, allowing the architecture to be extended to cloud environments later.

Example structure:

```text
infrastructure/
└── terraform/
    ├── providers.tf
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── terraform.tfvars.example
```

Development validation:

```bash
terraform init
terraform fmt -check
terraform validate
terraform plan
```

`terraform apply` is not required for this project because no permanent cloud infrastructure is required.

---

# 25. Phase 15 — Ansible

Ansible demonstrates configuration management for a hypothetical Linux/VM deployment environment.

Example:

```text
infrastructure/
└── ansible/
    ├── inventory/
    ├── playbooks/
    │   └── setup.yml
    └── roles/
        └── mlops_node/
```

A playbook may demonstrate configuration of:

```text
Docker
kubectl
Helm
Application directories
OS dependencies
Service configuration
```

Development validation:

```bash
ansible-playbook --syntax-check playbooks/setup.yml
```

Execution against production infrastructure is not required.

---

# 26. Repository Structure

A representative final repository structure is:

```text
ML-NLP-AgenticAI-MLOPS/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── rag/
│
├── src/
│   └── financial_ai/
│       │
│       ├── config/
│       │
│       ├── data/
│       │
│       ├── features/
│       │
│       ├── ml/
│       │
│       ├── nlp/
│       │
│       ├── rag/
│       │
│       ├── agents/
│       │
│       ├── api/
│       │
│       ├── monitoring/
│       │
│       └── scripts/
│
├── streamlit/
│
├── tests/
│
├── docker/
│
├── helm/
│   └── financial-ai/
│
├── infrastructure/
│   ├── terraform/
│   └── ansible/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
├── .github/
│   └── workflows/
│
├── docs/
│
├── requirements.txt
├── docker-compose.yml
├── .gitignore
└── README.md
```

The exact repository may differ as the project evolves.

---

# 27. Local Setup

## Clone Repository

```bash
git clone <repository-url>
cd ML-NLP-AgenticAI-MLOPS
```

## Create Environment

Using Conda:

```bash
conda create -n financial-ai python=<compatible-python-version>
conda activate financial-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure source path:

```bash
export PYTHONPATH=src
```

---

# 28. Data Versioning

Initialize DVC if required:

```bash
dvc init
```

Retrieve versioned datasets when a remote is configured:

```bash
dvc pull
```

Large datasets should not be committed directly to Git.

---

# 29. Running MLflow

Start MLflow:

```bash
mlflow ui --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Set the tracking URI when required:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

# 30. Running the ML Pipeline

Example:

```bash
python -m financial_ai.scripts.run_phase4
```

This can perform:

```text
Feature preparation
Target generation
Temporal splitting
Hyperparameter tuning
Model evaluation
MLflow logging
Model registration
```

---

# 31. Running the RAG Pipeline

Start Ollama if required:

```bash
ollama serve
```

Example model:

```bash
ollama pull qwen2.5:7b
```

Run RAG:

```bash
python -m financial_ai.scripts.run_phase5
```

The RAG workflow retrieves only filings eligible for the requested analysis date.

---

# 32. Running FastAPI

Example:

```bash
uvicorn financial_ai.api.main:app --reload
```

API documentation is available through FastAPI's generated Swagger interface.

Typical local location:

```text
http://127.0.0.1:8000/docs
```

---

# 33. Running Streamlit

Example:

```bash
streamlit run <streamlit-entry-file>
```

The Streamlit application communicates with the Financial AI backend and presents the combined ML, NLP, RAG and agentic analysis.

---

# 34. Testing

Run tests:

```bash
PYTHONPATH=src pytest -v
```

Important test categories include:

```text
Target generation
Temporal filtering
Point-in-time SEC retrieval
Feature engineering
Fundamental alignment
ML splits
RAG retrieval
API validation
```

Temporal tests are particularly important because preventing future-data leakage is a core project requirement.

---

# 35. Example End-to-End Workflow

A user requests:

```text
Ticker: AAPL
Analysis Date: 2026-07-15

"Analyze the company's current financial situation."
```

The system performs:

```text
                         Request
                            |
                            v
                       LangGraph
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
     ML Prediction      News NLP          SEC RAG
          |                 |                 |
          |                 |          latest eligible
          |                 |           10-Q / 8-K
          |                 |                 |
          +-----------------+-----------------+
                            |
                            v
                     Final Analysis
                            |
                            v
                        Streamlit
```

Crucially, all three branches respect the same:

```text
as_of_date
```

boundary.

---

# 36. MLOps Lifecycle

The intended lifecycle is:

```text
Raw Data
   |
   v
DVC
   |
   v
ETL / Validation
   |
   v
Feature Engineering
   |
   v
Model Training
   |
   v
Walk-Forward Validation
   |
   v
MLflow
   |
   v
Model Registry
   |
   v
Inference
   |
   +------------+
   |            |
   v            v
Evidently    Prometheus
   |            |
   |            v
   |         Grafana
   |
   v
Drift / Quality Detection
   |
   v
Retraining Decision
```

---

# 37. Development vs Production Scope

This repository is primarily a **development and portfolio implementation**.

Core application functionality is intended to be executable locally:

- data engineering
- ML
- NLP
- RAG
- LangGraph agents
- FastAPI
- Streamlit
- SHAP
- MLflow

Infrastructure and operations components demonstrate how the system could be productionized:

- Evidently monitoring
- Prometheus
- Grafana
- Docker
- Kubernetes
- Helm
- GitHub Actions
- Terraform
- Ansible

Not every infrastructure component is intended to represent an actively operated production deployment.

In particular:

> Terraform and Ansible are included as executable/reference templates. No permanent paid cloud infrastructure is required.

---

# 38. Key Engineering Challenges Addressed

This project goes beyond model training by addressing several practical ML engineering problems:

### Temporal Leakage

All sources are constrained by information availability.

### ML Validation

Time-series data uses walk-forward and purged validation instead of random splitting.

### Model Lifecycle

MLflow manages experiments and registered model versions.

### Reproducibility

DVC versions datasets and Docker provides reproducible runtime environments.

### RAG Quality

Hybrid retrieval and reranking improve SEC evidence selection.

### Hallucination Control

RAG generation is constrained to retrieved financial evidence.

### Prompt Injection

Retrieved documents are treated as untrusted context rather than executable instructions.

### Explainability

SHAP explains predictive model behavior.

### Observability

Model monitoring and system monitoring are separated using Evidently and Prometheus/Grafana.

### Modular AI

ML, NLP and RAG remain independent capabilities coordinated by LangGraph.

---

# 39. Current Project Status

```text
Phase 1   Environment / Git / DVC          Complete
Phase 2   Point-in-Time Data               Complete
Phase 3   NLP / FinBERT                    Complete
Phase 4   ML / MLflow                      Complete
Phase 5   SEC RAG                          Complete
Phase 6   LangGraph Agentic AI             Complete
Phase 7   FastAPI                          Complete
Phase 8   Streamlit                        Complete
Phase 9   SHAP Explainability              Implementation
Phase 10  Monitoring                       Dev Templates
Phase 11  Security                         Dev Implementation
Phase 12  Docker                           Dev Deployment
Phase 13  Kubernetes / Helm                Dev Templates
Phase 14  GitHub Actions                   CI/CD Templates
Phase 15  Terraform / Ansible              IaC Templates
Phase 16  Documentation                    Finalization
```

---

# 40. Future Improvements

Potential future extensions include:

- additional equities
- multi-company analysis
- portfolio-level analysis
- improved SEC XBRL normalization
- additional macroeconomic data
- event classification
- advanced RAG evaluation
- automated retraining triggers
- model promotion workflows
- distributed inference
- cloud deployment
- managed Kubernetes
- model serving infrastructure
- feature store
- online monitoring
- human-in-the-loop agent workflows

These are intentionally outside the initial development scope.

---

# 41. Disclaimer

This project is intended for:

- educational purposes
- research
- software engineering demonstrations
- machine-learning experimentation

It is **not financial or investment advice**.

Predictions and generated analyses should not be used as the sole basis for investment decisions.

---

# 42. Summary

This project demonstrates an end-to-end Financial AI architecture combining:

```text
Data Engineering
      +
Machine Learning
      +
NLP
      +
RAG
      +
Agentic AI
      +
Explainable AI
      +
MLOps
      +
Monitoring
      +
Security
      +
DevOps
```

The central engineering principle is:

> **Generate financial analysis using only information that was actually available at the requested point in time.**

This enables the ML, NLP, RAG and agentic components to operate under a common temporal boundary while remaining independently testable and maintainable.