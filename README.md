# Multi-Agent E-commerce AI Analyst

A multi-agent AI system for e-commerce product analysis that combines machine learning, retrieval, and explainable AI.  
The project predicts product price class using structured and text-based features, and will later explain predictions using RAG-based evidence retrieval and multi-agent workflows.

## Features
- Product-level data pipeline from Amazon Reviews 2023 (Electronics)
- Baseline ML model for price class prediction
- Improved ML model with TF-IDF text features
- Saved model artifacts and prediction pipeline
- Upcoming RAG-based evidence retrieval
- Upcoming multi-agent analysis workflow
- Evaluation of ML and RAG outputs

## Architecture
- Data Agent → loads product and review data
- Forecast Agent → predicts product price class
- Retrieval Agent → retrieves supporting evidence
- Report Agent → generates grounded explanation
- Guardrail Agent → validates output consistency

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- FAISS / Chroma (planned)
- LLM API (planned)

## Project Structure
- `app/data/` → loading, cleaning, merging, feature engineering
- `app/models/forecasting/` → training and prediction
- `app/rag/` → retrieval pipeline
- `app/agents/` → multi-agent workflow
- `app/evaluation/` → ML and RAG evaluation
- `artifacts/models/` → trained model files

## Phase 1: Data Pipeline ✅

Completed:
- Downloaded and sampled Amazon Reviews 2023 (Electronics)
- Built data loaders for reviews and metadata
- Cleaned and normalized datasets
- Merged reviews with product metadata
- Built product-level feature table

Outputs:
- ~50K cleaned reviews
- ~34K matched products
- Product-level feature dataset for ML modeling

## Phase 2: ML Baseline and Improved Text Model ✅

Completed:
- Created supervised labels using price tertiles (`low`, `mid`, `high`)
- Trained a baseline numeric-feature model
- Trained an improved text-enhanced model using TF-IDF on title and category fields
- Saved trained models and label encoders
- Built prediction script for inference

Key results:
- Baseline numeric model accuracy: **0.38**
- Text-enhanced model accuracy: **0.71**

Main insight:
- Structured review statistics alone were weak predictors of price class
- Semantic text features dramatically improved performance
- This motivates the next phase: retrieval and explanation with RAG

## Phase 3A: RAG Retrieval Pipeline ✅

Completed:
- Built review document corpus for retrieval
- Generated sentence embeddings using MiniLM
- Created FAISS vector index
- Implemented global semantic retrieval
- Implemented product-specific semantic retrieval
- Built RAG service for structured evidence extraction
- Integrated ML prediction + RAG evidence in analysis service

Current analyst output:
- product metadata
- predicted price class
- retrieved supporting review evidence

Saved artifacts:
- `artifacts/models/price_class_model.joblib`
- `artifacts/models/price_class_label_encoder.joblib`
- `artifacts/models/price_class_model_with_text.joblib`
- `artifacts/models/price_class_label_encoder_with_text.joblib`

## Evaluation Layer

Implemented:
- Agent pipeline evaluation
- Retrieval quality checks
- Report alignment checks

Evaluation scripts:
- `python -m app.evaluation.agent_eval`
- `python -m app.evaluation.rag_eval`
- `python -m app.evaluation.report_eval`
- `python -m app.evaluation.run_all_eval`

## Current Status
✅ Phase 1 complete  
✅ Phase 2 complete  
🔄 Phase 3 next: RAG pipeline for evidence retrieval and explanation

## How to Run

### Phase 1 preprocessing
```bash
python -m app.data.preprocessing.clean_reviews
python -m app.data.preprocessing.clean_metadata
python -m app.data.preprocessing.merge_data
python -m app.data.preprocessing.build_features
python -m app.data.preprocessing.create_labels