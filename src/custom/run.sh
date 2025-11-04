#!/bin/bash

export PYTHONPATH=$PWD && uv run python -m src.custom.run \
  --input-name PL-DR-DP-BK-01 \
  --dataset-name PL-DR-DP-BK-01_10_0 \
  --kg-name PL-DR-DP-BK-01 \
  --testset-size 10 \
  --llm-name gemini-2.0-flash \
  --embedding-model-name "models/text-embedding-004" \
  --languages en th \
  --num-personas 3 \
  --persona-file-path "cache/persona.json" \
  --single-hop-probability 0.7 \
  --multi-hop-probability 0.3 \
  --multi-hop-abstract-probability 0
