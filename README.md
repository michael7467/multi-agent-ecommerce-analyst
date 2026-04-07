# Multi-Agent E-commerce AI Analyst

A multi-agent AI system that predicts product price movement and explains predictions using RAG-based evidence retrieval.

## Features
- Price trend prediction (ML model)
- Retrieval-Augmented Generation (RAG)
- Multi-agent architecture
- Evaluation of ML and RAG outputs

## Architecture
- Data Agent → loads product + reviews
- Forecast Agent → predicts trend
- Retrieval Agent → retrieves evidence
- Report Agent → generates explanation

## Tech Stack
- Python
- FastAPI
- LightGBM / XGBoost
- FAISS / Chroma
- LLM API

## Phase 1: Data Pipeline ✅

Completed:
- Downloaded and sampled Amazon Reviews 2023 (Electronics)
- Built data loaders for reviews and metadata
- Cleaned and normalized datasets
- Merged reviews with product metadata
- Built product-level feature table

Final dataset:
- ~50K reviews
- ~34K products
- Product-level features for ML modeling