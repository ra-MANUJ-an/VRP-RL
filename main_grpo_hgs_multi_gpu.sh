set -x

export WANDB_API_KEY="9de25c4a8eed15d718cdf323a46ba18ad28aebb7"

PROJECT_NAME="GRPO_HGS"
RUN_NAME="Qwen2.5-Coder-7B-Instruct-$(date '+%m%d')"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/hdfs/zhangcong/tsp/train_20250509.parquet \
    data.val_files=/mnt/hdfs/zhangcong/tsp/test_20250509.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/mnt/hdfs/codeai/hf_models/Qwen2.5-Coder-7B-Instruct\
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/mnt/hdfs/zhangcong/CodeAI/verl_rl_checkpoints \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=-1 \
    trainer.test_freq=100 \
    +trainer.val_before_train=False \
    trainer.total_epochs=100 $@ 2>&1 | tee grpo.log
