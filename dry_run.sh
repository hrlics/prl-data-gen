bash run_genvf_v10.sh \
    debug.mode=actor finetune.attempts=1 actor.llm_max_rollouts=1 finetune.push_to_hub=false \
    llm.wandb_table.log_every_n_groups=1 llm.wandb_table.keep_last_k_groups=32 \
    llm_grader.wandb_table.log_every_n_groups=1 llm_grader.wandb_table.keep_last_k_groups=32 > check.log 2>&1