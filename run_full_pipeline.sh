#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline: infrastructure -> data pipeline -> embeddings/
# index -> MLflow training -> eval. Run from the project root, same
# directory as docker-compose.yml.
#
# HONEST CONFIDENCE MARKERS, not uniform certainty: paths below marked
# [CONFIRMED] were directly verified this session (imports, Dockerfile
# CMD lines, or actual test runs). Paths marked [UNCONFIRMED] are
# reconstructed from what these scripts were understood to do earlier
# in this session, but their exact module paths were never directly
# re-verified -- edit the variables below if they don't match your
# actual layout, rather than assume the script is wrong.

# ---- Configurable paths (edit if these don't match your repo) ----
DATA_PIPELINE_MODULES=(
    # [UNCONFIRMED] -- reviewed early in this session, exact paths not
    # re-verified since. Edit these to match your actual scripts/ or
    # app/data/ layout before relying on this section.
    "app.data.loaders.reviews_loader"
    "app.data.loaders.metadata_loader"
    "app.data.preprocessing.clean_reviews"
    "app.data.preprocessing.clean_metadata"
    "app.data.preprocessing.merge_data"
    "app.data.preprocessing.build_features"
    "app.data.preprocessing.create_labels"
    "app.data.preprocessing.build_sentiment_features"
)
TRAINING_MODULE="app.models.forecast.train_text_model_mlflow"  # [UNCONFIRMED] -- delivered standalone, never placed into your actual tree

CHUNKING_MODULE="app.rag.chunking"                              # [CONFIRMED]
EMBED_REVIEWS_MODULE="app.models.embeddings.embed_reviews"      # [CONFIRMED]
EMBED_IMAGES_MODULE="app.models.embeddings.embed_images"        # [CONFIRMED]
QDRANT_INDEX_MODULE="app.rag.qdrant_index_builder"               # [CONFIRMED]
EVAL_MODULE="app.evaluation.runners.run_all_eval"                # [CONFIRMED]

# ---- Flags ----
SKIP_INFRA=false
SKIP_DATA=false
SKIP_TRAINING=false
SKIP_EVAL=false

for arg in "$@"; do
    case "$arg" in
        --skip-infra) SKIP_INFRA=true ;;
        --skip-data) SKIP_DATA=true ;;
        --skip-training) SKIP_TRAINING=true ;;
        --skip-eval) SKIP_EVAL=true ;;
        --help)
            echo "Usage: $0 [--skip-infra] [--skip-data] [--skip-training] [--skip-eval]"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg (use --help)"
            exit 1
            ;;
    esac
done

log() { echo; echo "==> $1"; echo; }

# ---- Step 1: infrastructure ----
if [ "$SKIP_INFRA" = false ]; then
    log "Step 1/5: Starting infrastructure (api, streamlit, qdrant, redis, otel-collector, prometheus, grafana)"

    if [ -z "${PROMETHEUS_BEARER_TOKEN:-}" ] || [ -z "${GRAFANA_ADMIN_PASSWORD:-}" ]; then
        echo "PROMETHEUS_BEARER_TOKEN and GRAFANA_ADMIN_PASSWORD must be set -- see generate-prometheus-config.sh"
        exit 1
    fi
    ./generate-prometheus-config.sh

    # --wait blocks until every service with a healthcheck reports
    # healthy, not just started -- confirmed directly before writing
    # this, since the later steps genuinely depend on qdrant actually
    # being ready, not merely launched. --wait-timeout prevents this
    # script hanging forever if something never becomes healthy.
    docker compose up -d --build --wait --wait-timeout 180
else
    log "Step 1/5: Skipped (--skip-infra) -- assuming infrastructure is already running"
fi

# ---- Step 2: data pipeline ----
if [ "$SKIP_DATA" = false ]; then
    log "Step 2/5: Data pipeline"
    for module in "${DATA_PIPELINE_MODULES[@]}"; do
        echo "  Running $module ..."
        docker compose exec -T api python -m "$module"
    done
else
    log "Step 2/5: Skipped (--skip-data)"
fi

# ---- Step 3: embeddings + index build ----
if [ "$SKIP_DATA" = false ]; then
    log "Step 3/5: Embeddings and Qdrant index (hybrid dense+sparse)"
    docker compose exec -T api python -m "$CHUNKING_MODULE"
    docker compose exec -T api python -m "$EMBED_REVIEWS_MODULE"
    docker compose exec -T api python -m "$EMBED_IMAGES_MODULE"
    docker compose exec -T api python -m "$QDRANT_INDEX_MODULE"
else
    log "Step 3/5: Skipped (--skip-data)"
fi

# ---- Step 4: MLflow training ----
if [ "$SKIP_TRAINING" = false ]; then
    log "Step 4/5: MLflow-tracked training (skops serialization)"
    docker compose exec -T api python -m "$TRAINING_MODULE"
else
    log "Step 4/5: Skipped (--skip-training)"
fi

# ---- Step 5: eval ----
if [ "$SKIP_EVAL" = false ]; then
    log "Step 5/5: Running evals against the live orchestrator"
    docker compose exec -T api python -m "$EVAL_MODULE"
else
    log "Step 5/5: Skipped (--skip-eval)"
fi

log "Pipeline complete."