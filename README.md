🧠 Multi‑Agent E‑Commerce Analyst
AI‑powered product intelligence platform with multi‑agent orchestration, RAG, sentiment analysis, topic modeling, and real‑time insights.

A fully production‑ready system built with:

FastAPI (backend API)

Streamlit (UI)

Multi‑Agent Orchestration (Planner, Data, Memory, Report, Critic, Guardrail…)

Redis (caching + message bus)

Qdrant (vector search for RAG + embeddings)

OpenTelemetry (tracing + metrics)

Docker + Kubernetes (deployment)

GitHub Actions (CI/CD)

This platform analyzes e‑commerce products using LLM‑driven agents, retrieves evidence from reviews, generates insights, and produces a final decision report.

🚀 Features
🔹 Multi‑Agent Architecture
A coordinated set of agents:

PlanningAgent — builds the execution plan

DataAgent — fetches product data + reviews

SentimentAgent — global + aspect sentiment

TopicAgent — pain‑point detection + theme extraction

RetrievalAgent — RAG over review embeddings

ImageRetrievalAgent — multimodal similarity

CounterfactualAgent — “what would improve this product?”

ReportAgent — final structured report

GuardrailAgent — safety + hallucination checks

CriticAgent — evaluates agent outputs

All orchestrated through a robust, fault‑tolerant pipeline.

🧩 System Architecture
High‑Level Overview
Code
                ┌──────────────────────────┐
                │        Streamlit UI       │
                └──────────────┬───────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │      FastAPI      │
                    │   Multi‑Agent     │
                    └─────────┬─────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
Redis Cache            Qdrant Vector DB         OpenTelemetry
Caching, rate‑limiting   Embeddings + RAG       Traces + Metrics
🛠️ Tech Stack
Backend
FastAPI

Python 3.12

Pydantic

Async orchestration

OpenTelemetry instrumentation

Frontend
Streamlit

Real‑time product analysis dashboard

AI / ML
LLM‑based agents

RAG with Qdrant

Sentiment + aspect sentiment

Topic modeling

Counterfactual reasoning

Infrastructure
Docker

Kubernetes (Deployments, StatefulSets, Services, Ingress)

Redis (cache)

Qdrant (vector DB)

GitHub Actions CI/CD

📦 Project Structure
Code
app/
  ├── api/
  │   ├── main.py
  │   ├── routers/
  │   └── services/
  ├── agents/
  │   ├── planning_agent.py
  │   ├── data_agent.py
  │   ├── sentiment_agent.py
  │   ├── topic_agent.py
  │   ├── retrieval_agent.py
  │   ├── report_agent.py
  │   └── ...
  ├── core/
  ├── models/
  ├── utils/
  └── ui/
      └── streamlit_app.py
k8s/
  ├── api-deployment.yaml
  ├── api-service.yaml
  ├── streamlit-deployment.yaml
  ├── streamlit-service.yaml
  ├── redis-deployment.yaml
  ├── redis-service.yaml
  ├── qdrant-statefulset.yaml
  ├── qdrant-service.yaml
  ├── ingress.yaml
  ├── configmap.yaml
  └── secret.yaml
🐳 Running Locally (Docker Compose)
bash
docker-compose up --build
Services:

API → http://localhost:8000

Streamlit UI → http://localhost:8501

Redis → localhost:6379

Qdrant → http://localhost:6333

☸️ Kubernetes Deployment
Apply all manifests:

bash
kubectl apply -f k8s/
Access the UI via Ingress:
Code
https://ui.yourdomain.com
Access the API:
Code
https://api.yourdomain.com
🔐 Environment Variables
ConfigMap (ecommerce-config)
ENV

LOG_LEVEL

REDIS_URL

QDRANT_URL

METRICS_PORT

RATE_LIMIT_PER_MINUTE

OTEL_*

Secret (ecommerce-secrets)
OPENAI_API_KEY

QDRANT_API_KEY

API_KEY

📊 Observability
The system includes:

OpenTelemetry tracing

Metrics endpoint

Health, readiness, and liveness probes

Distributed tracing across agents

You can plug this into:

Grafana

Tempo

Prometheus

Jaeger

🧪 Testing
Run unit tests:

bash
pytest -q
📈 Roadmap
[ ] Add multi‑product batch analysis

[ ] Add user authentication

[ ] Add product comparison agent

[ ] Add fine‑tuned embedding model

[ ] Add GPU support for inference

🤝 Contributing
Pull requests are welcome.
Please open an issue first to discuss major changes.

📄 License
MIT License.

🙌 Acknowledgements
This project integrates:

Qdrant (vector search)

Redis (caching)

Streamlit (UI)

FastAPI (backend)

OpenTelemetry (observability)