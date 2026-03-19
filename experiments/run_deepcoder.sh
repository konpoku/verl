set -x

# Data path: where prepare_deepcoder_data.py saved the parquet files
# Override with: DEEPCODER_DATA_DIR=/your/path bash run_deepcoder.sh
DATA_DIR=${DEEPCODER_DATA_DIR:-/opt/tiger/deepcoder}
train_files=${DATA_DIR}/train.parquet
test_files=${DATA_DIR}/test.parquet

# Base model for actor, rollout and critic
# For code tasks a coder model is recommended, e.g.:
#   Qwen/Qwen2.5-Coder-7B-Instruct
#   deepseek-ai/deepseek-coder-7b-instruct-v1.5
#MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}
MODEL_PATH=Qwen/Qwen3-14b

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    reward.num_workers=16 \
    reward.reward_model.enable=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='deepcoder_ppo' \
    trainer.experiment_name='deepcoder_code_reward' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/workspace/data/checkpoints \
    trainer.rollout_data_dir=/workspace/data/outputs $@
