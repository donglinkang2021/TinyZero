#!/bin/bash

# List of models to download
models=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-Math-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # base Qwen2.5-Math-1.5B
    "/data1/linkdom/hf_models/Open-RS1" # base DeepSeek-R1-Distill-Qwen-1.5B
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-Coder-1.5B"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-3B"
    "Qwen/Qwen2.5-Coder-3B-Instruct"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-Math-7B"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # base Qwen2.5-Math-7B
    "GD-ML/Qwen2.5-Math-7B-GPG" # base Qwen2.5-Math-7B
    "Qwen/Qwen2.5-Coder-7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
)

# Loop through the models and download each one
for model_name in "${models[@]}"; do
    echo "Downloading ${model_name}..."
    python download_model.py --model_name "${model_name}"
    echo "Finished downloading ${model_name}."
    echo "-------------------------------------"
done

echo "All specified models downloaded."