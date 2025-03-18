
# VRP-RL

---

## Installation

```bash
conda create -n logic python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib
```

---

## Data Preparation

You can directly use /data.

For your own data generation, here's a demo:

### Base Model
```bash
python ./examples/data_preprocess/tsp20.py \
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

### Instruct Model
```bash
python ./examples/data_preprocess/tsp20.py \
    --template_type=qwen-instruct \
    --local_dir {processed_data_path} \
    --data_path {raw_data_path}
```

---

## Training Execution
```bash
conda activate logic
bash main_grpo.sh  # 4×A100 80G
```

---

## ⚙️ Implementation Details

| Component              | Location                          |
|------------------------|-----------------------------------|
| Reward Modeling     | `verl/utils/reward_score/tsp20.py`   |
| Data Preprocessing   | `examples/data_preprocess/tsp20.py`  |

---


## Citation
Coming Soon

---

## Acknowledgements
- [Verl](https://github.com/volcengine/verl) 🔗
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 🔗
- [Logic-RL](https://github.com/Unakar/Logic-RL) 🔗
- [Knights and Knaves (K&K) puzzles dataset](https://github.com/AlphaPav/mem-kk-logic) 🔗

---
