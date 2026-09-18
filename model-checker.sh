#!/bin/bash

model_list=./config/ollama-models.txt
existing_models=$(ollama list | awk 'NR>1 {print $1}')

for model in $(grep -vE '^\s*$|^\s*#' $model_list); do

  if echo "$existing_models" | grep -Fxq "$model"; then
    echo "[PASS] $model"
  else
    ollama pull "$model" && echo "[SUCCESS] $model"

  fi
done