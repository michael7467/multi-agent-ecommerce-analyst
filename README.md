# 🧠 Multi-Agent E-Commerce Analyst

AI-powered product intelligence platform with multi-agent orchestration, RAG, sentiment analysis, topic modeling, and real-time insights.

A production-ready system that analyzes e-commerce products using LLM-driven agents, retrieves evidence from reviews, generates insights, and produces a final decision report.

---

## 🚀 Overview

This platform combines modern AI techniques with scalable backend architecture to deliver deep product intelligence:

- 🔍 Extract insights from product reviews  
- 💬 Perform sentiment & aspect-level analysis  
- 🧠 Use Retrieval-Augmented Generation (RAG) for grounded answers  
- 📊 Generate structured decision reports  
- ⚙️ Operate through a coordinated multi-agent pipeline  

---

## ✨ Features

### 🔹 Multi-Agent Architecture

A coordinated system of specialized agents:

| Agent | Responsibility |
|------|----------------|
| PlanningAgent | Builds execution plan |
| DataAgent | Fetches product data & reviews |
| SentimentAgent | Global + aspect sentiment analysis |
| TopicAgent | Extracts themes & pain points |
| RetrievalAgent | RAG over review embeddings |
| ImageRetrievalAgent | Multimodal similarity |
| CounterfactualAgent | Suggests product improvements |
| ReportAgent | Generates final structured report |
| GuardrailAgent | Safety & hallucination checks |
| CriticAgent | Evaluates agent outputs |

All agents are orchestrated in a fault-tolerant pipeline.

---

## 🧩 System Architecture

```mermaid
flowchart TD
    UI[Streamlit UI] --> API[FastAPI Multi-Agent System]
    API --> Redis[Redis Cache]
    API --> Qdrant[Qdrant Vector DB]
    API --> OTEL[OpenTelemetry]

    subgraph Agents
        Planner[PlanningAgent]
        Data[DataAgent]
        Sentiment[SentimentAgent]
        Topic[TopicAgent]
        Retrieval[RetrievalAgent]
        Image[ImageRetrievalAgent]
        Counter[CounterfactualAgent]
        Report[ReportAgent]
        Guard[GuardrailAgent]
        Critic[CriticAgent]
    end
    API --> Planner
    Planner --> Data
    Data --> Sentiment
    Sentiment --> Topic
    Topic --> Retrieval
    Retrieval --> Image
    Image --> Counter
    Counter --> Report
    Report --> Guard
    Guard --> Critic
` ` `
🛠️ Tech Stack
🔧 Backend
FastAPI
Python 3.12
Pydantic
Async orchestration
OpenTelemetry instrumentation
🎨 Frontend
Streamlit
Real-time analysis dashboard
🤖 AI / ML
LLM-based agents
RAG with Qdrant
Sentiment analysis (global + aspect)
Topic modeling
Counterfactual reasoning
☁️ Infrastructure
Docker
Kubernetes (Deployments, StatefulSets, Services, Ingress)
Redis (caching & messaging)
Qdrant (vector database)
GitHub Actions (CI/CD)
📦 Project Structure
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
docker-compose up --build
Services
Service	URL
API	http://localhost:8000

Streamlit UI	http://localhost:8501

Redis	localhost:6379
Qdrant	http://localhost:6333
☸️ Kubernetes Deployment

Apply all manifests:

kubectl apply -f k8s/
Access
UI → https://ui.yourdomain.com
API → https://api.yourdomain.com
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

Compatible with:

Grafana
Tempo
Prometheus
Jaeger
🧪 Testing
pytest -q
📈 Roadmap
 Multi-product batch analysis
 User authentication
 Product comparison agent
 Fine-tuned embedding model
 GPU support for inference
🤝 Contributing

Pull requests are welcome.
Please open an issue first to discuss major changes.

📄 License

MIT License.

🙌 Acknowledgements
Qdrant — vector search
Redis — caching & messaging
Streamlit — UI
FastAPI — backend API
OpenTelemetry — observability