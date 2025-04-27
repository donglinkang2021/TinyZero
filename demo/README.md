# README

Record of my simple demo `.py` scripts for the TinyZero project. The scripts are used to

1. [Download the pretrained model](./download_model.py)
2. [Test vllm](./openai_demo.py)

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 4
```
