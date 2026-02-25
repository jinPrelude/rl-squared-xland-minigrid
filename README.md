# RL² on XLand-MiniGrid

Minimal [RL²](https://arxiv.org/abs/1611.02779) implementation using [Flax NNX](https://github.com/google/flax) on [XLand-MiniGrid](https://arxiv.org/abs/2312.12044).

## Installation

```bash
conda create -n rl2-xland python=3.12 -y
conda activate rl2-xland
pip install -r requirements.txt

# GPU: install JAX with proper CUDA version (e.g. CUDA 12)
pip install "jax[cuda12]"
```

## Usage

**1x A100** (original setup from XLand-MiniGrid paper)

```bash
python main.py --model gru --benchmark-id small-1m --num-envs 16384 --num-minibatches 32
```

**1x L40S** — batch size 512 → 256, gradient steps halved (32 → 16)

```bash
python main.py --model gru --benchmark-id small-1m --num-envs 1024 --num-minibatches 4 --num-epochs 4
```
