#!/usr/bin/env bash
set -euo pipefail

# Generates the real prometheus.yml from the template, substituting the
# admin token from the environment -- run on the host, before any
# container starts, specifically to avoid depending on whatever tooling
# may or may not be available inside the official Prometheus image
# (a minimal, unprivileged busybox variant with no confirmed shell/
# envsubst access). This way Prometheus itself stays completely stock,
# just reading an already-finished config file.

: "${PROMETHEUS_BEARER_TOKEN:?PROMETHEUS_BEARER_TOKEN must be set -- use an admin_api_keys value}"

sed "s|\${PROMETHEUS_BEARER_TOKEN}|${PROMETHEUS_BEARER_TOKEN}|g" \
    prometheus.yml.template > prometheus.yml

echo "Generated prometheus.yml"
