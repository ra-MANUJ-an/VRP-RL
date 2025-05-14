```bash
# Setup env
cd /opt/tiger/verl_code_ai_utils
bash setup.sh
git clone --branch zhangcong_devrp git@github.com:SivilTaram/verl.git
cd /opt/tiger/verl_code_ai_utils/verl
bash install.sh
pip install "bytedray[default,data,serve,bytedance]"~=2.10.0.0
pip install pydantic==2.10.6
pip install wandb IPython matplotlib
pip install meson ninja
cd /opt/tiger/verl_code_ai_utils
rm -rf verl
unzip tsp_100_instances.zip -d tsp_100_instances
git clone --branch zhangcong_dev_based_on_munuj_0514  git@github.com:ra-MANUJ-an/VRP-RL.git
cd /opt/tiger/verl_code_ai_utils/VRP-RL/PyVRP
pip install -e . -U

export PROJECT_NAME=GRPO_HGS
export RUN_NAME=$(date '+%m%d')
export WANDB_API_KEY="9de25c4a8eed15d718cdf323a46ba18ad28aebb7"
export WANDB_OFFICIAL=1
export HDFS_DATA_PATH="/mnt/hdfs/zhangcong/"
export HDFS_MODEL_PATH="/mnt/hdfs/codeai/hf_models"
export HDFS_CHECKPOINT_PATH="/mnt/hdfs/zhangcong/CodeAI/verl_rl_checkpoints"
export HDFS_LOG_PATH="/mnt/hdfs/zhangcong/CodeAI/verl_rl_logs"
export MAX_RUNTIME_HOURS=0
export VLLM_USE_V1=1

# Train
cd /opt/tiger/verl_code_ai_utils/VRP-RL
bash main_grpo_hgs_multi_gpu.sh
```