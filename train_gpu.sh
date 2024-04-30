# DEBUG=true 
# accelerate launch --num_processes=1 --mixed_precision='bf16' \
# accelerate launch --config_file='/infini/fsdb_config.yaml' \
accelerate launch --num_processes=1 --mixed_precision='bf16' \
    train.llama.infini.noclm.py \
    --segment_length=128 \
    --block_size=512 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir='./models/tiny-llama-infini-noclm-wikitext' \
    --checkpointing_steps=20 \
    --num_train_epochs=1 \
    --learning_rate=1e-5 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=6 \
    --with_tracking \
    --max_train_steps=50 \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --model_name_or_path='openlm-research/open_llama_3b_v2' \
    # --model_name_or_path='openlm-research/open_llama_3b_v2' \
    # --model_name_or_path='reciprocate/tiny-llama' \
    # --resume_from_checkpoint='./models/tiny-llama-infini-noclm-wikitext/step_5' \
    # --dataset_name='wikitext' \
    # --dataset_config_name='wikitext-2-raw-v1' \
    # --push_to_hub \
    # --hub_model_id='kerinin/infini' \
    # --dataset_name='JeanKaddour/minipile' \
    # --output_dir='./models/llama-3-8b-infini-noclm-minipile' \
    # --model_name_or_path='meta-llama/Meta-Llama-3-8B' \
    
    #--multi_gpu --num_processes=2
#accelerate launch --config_file='/infini/fsdb_config.yaml' \
    # --block_size=1048576 \

# accelerate launch --mixed_precision='bf16' \
#     train.gemma.infini.noclm.py \
#     --model_name_or_path='google/gemma-2b' \
#     --segment_length=2048 \
#     --block_size=32768 \
#     --dataset_name='wikitext' \
#     --dataset_config_name='wikitext-2-raw-v1' \
#     --per_device_train_batch_size=2 \
#     --per_device_eval_batch_size=2 \
#     --weight_decay=1.0 \
#     --output_dir='./models/gemma-2b-infini-noclm-wikitext' \
#     --num_train_epochs=1 \
#     --learning_rate=5e-5 \
#     --seed=42 \
#     --low_cpu_mem_usage \
#     --report_to='wandb' \
#     --preprocessing_num_workers=64 \
#     --with_tracking \
#     --push_to_hub \
#     --hub_model_id='kerinin/infini' \
