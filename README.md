# TinyZero
![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own 

You can experience the Ahah moment yourself for < $30 

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

Paper's on it's way!

## Installation

```bash
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install ray

# verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Huggingface settings

Before you run any script here, please set this two enviroment variable here:

```bash
# export HF_HOME=/root/autodl-tmp/.cache/huggingface # autodl
export HF_HOME=/data1/linkdom/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
# echo $HF_HOME
# du -h --max-depth=1 $HF_HOME/hub
# check you qwen model here
du -h --max-depth=1 $HF_HOME/hub | grep Qwen
```

## Countdown task

**Data Preparation**

```bash
conda activate zero
# python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
python ./examples/data_preprocess/countdown.py --local_dir /data1/linkdom/zero/data/countdown
# python ./examples/data_preprocess/countdown.py --local_dir /root/autodl-tmp/zero/data/countdown # autodl
```

## Download Pretrained Model

See [here](./demo/download_model.py).

### Run Training

```bash
conda activate zero
wandb login
<view https://wandb.ai/authorize to get your_wandb_key>
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script, and checkout the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643)

**Single GPU**

Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```bash
export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=/data1/linkdom/zero/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

> [!NOTE] Comment
> - 20250324_1707: Qwen2.5-0.5B has 14 heads should not set rollout_tp_size to 4, 就是说14不能被4整除，只能是2，而且实测发现45GB的空间依然不够，目前思考的方向是去找更大的机器跑一下
> - 20250324_1858: 实测成功了，在autodl A800x1 80G上跑的，修改了batch_size为64才终于能跑，最高能到50G占用，起码是跑通了，但是发现模型一直学不到东西，按照之前的思路，拿math 1.5B的模型来跑，看看能不能学到东西

```bash
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-Math-1.5B
export DATA_DIR=/data1/linkdom/zero/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-math-1.5b
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1
bash ./scripts/train_tiny_zero.sh
```

```bash
export N_GPUS=2
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=/data1/linkdom/zero/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=2,3
bash ./scripts/train_tiny_zero.sh
```

> [!NOTE] Comment
> - 20250324_2056: 基本跑通了第一轮看到reponse的长度在不断上升，获得的奖励也变得更好了，但还是没完整跑完一轮，现在是充分利用好4xA6000来跑下看结果，其中跑math的两个卡显存都拉到了40G，而跑0.5B的显存维持在每个卡15G以下，打算明天跑完看下结果：感觉这个框架还是有点复杂的。

**3B+ model**

In this case, the base model is able to develop sophisticated reasoning skills.

```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation

We experiment with QWen-2.5-3B Instruct too.

**Data Preparation**

To follow chat template, we need to reprocess the data:

```bash
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**

```bash
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledge

* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation

```bibtex
@misc{tinyzero,
    author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
    title        = {TinyZero},
    howpublished = {https://github.com/Jiayi-Pan/TinyZero},
    note         = {Accessed: 2025-01-24},
    year         = {2025}
}
```
