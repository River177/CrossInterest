#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-8B}"
: "${LORA_ADAPTER_PATH:?LORA_ADAPTER_PATH is required}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${CHECKPOINT_DIR:=$OUTPUT_DIR/checkpoints}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${N_GPUS:=4}"
: "${TRAIN_MAX_SAMPLES:=-1}"
: "${TRAIN_BATCH_SIZE:=32}"
: "${DATALOADER_NUM_WORKERS:=0}"
: "${MAX_PROMPT_LENGTH:=2048}"
: "${MAX_RESPONSE_LENGTH:=768}"
: "${TOTAL_EPOCHS:=1}"
: "${TOTAL_STEPS:=null}"
: "${LR:=1e-5}"
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${ROLLOUT_N:=16}"
: "${ROLLOUT_TP:=1}"
: "${ROLLOUT_GPU_MEMORY:=0.4}"
: "${PPO_MINI_BATCH_SIZE:=32}"
: "${PPO_MICRO_BATCH_SIZE:=1}"
: "${SEED:=42}"
: "${SAVE_FREQ:=25}"
: "${MAX_ACTOR_CKPT_TO_KEEP:=2}"
: "${TEST_FREQ:=-1}"
: "${AGENT_LOOP_WORKERS:=8}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$TRAIN_FILE" \
  "data.train_max_samples=$TRAIN_MAX_SAMPLES" \
  data.val_max_samples=1 \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
  "data.max_response_length=$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS" \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  "actor_rollout_ref.model.lora_rank=$LORA_RANK" \
  "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  "actor_rollout_ref.actor.optim.lr=$LR" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  "actor_rollout_ref.rollout.temperature=1.0" \
  "actor_rollout_ref.rollout.top_p=0.95" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY" \
  "actor_rollout_ref.rollout.n=$ROLLOUT_N" \
  "actor_rollout_ref.rollout.agent.num_workers=$AGENT_LOOP_WORKERS" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  "custom_reward_function.path=$PROJECT_DIR/src/new_interest/grpo_reward.py" \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive \
  "+ray_kwargs.ray_init.num_gpus=$N_GPUS" \
  "trainer.n_gpus_per_node=$N_GPUS" \
  trainer.nnodes=1 \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.total_training_steps=$TOTAL_STEPS" \
  trainer.val_before_train=False \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.test_freq=$TEST_FREQ" \
  trainer.logger=wandb \
  trainer.project_name=cross-interest-generative-grpo \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.default_local_dir=$CHECKPOINT_DIR" \
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.resume_mode=auto \
  "$@"
