#!/bin/bash

exp_tag="finetune"
python finetune.py \
    --base_model '/root/ours/hf_llama-med-pretraining_en_zh' \
    --data_path './data/training_data(en+zh).json' \
    --output_dir './lora-llama-med-'$exp_tag \
    --prompt_template_name 'alpaca' \
    --micro_batch_size 256 \
    --batch_size 256 \
    --wandb_run_name $exp_tag \