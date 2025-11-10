#! /bin/bash

# This is the example script to pretrain a 45M LLaMA model with AdamW

#SBATCH -J adam-45m
#SBATCH -o printouts/adam-45m/%A_%a.out
#SBATCH -e printouts/adam-45m/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=1
#SBATCH -p kempner_h100
#SBATCH --account=kempner_barak_lab
#SBATCH --constraint="h100"
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=370G
#SBATCH -c 24


# Put your WANDB API key here to enable logging to wandb.
export PYTHONPATH="${PWD}:$PYTHONPATH"
export JAX_TRACEBACK_FILTERING=off
export WANDB_API_KEY=


module load python/3.10.12-fasrc01;
mamba activate EasyLM;


# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

# For baselines: total_steps, log_freq, eval_freq, save_model_freq count each grad acc step as 1 step
# lr_warmup_steps, lr_decay_steps count *optimizer* steps (total_steps // grad_acc)

python sweep_launcher.py \
    --program='EasyLM.models.llama.llama_train' \
    --load_checkpoint='trainstate_params::/n/netscratch/kempner_barak_lab/Everyone/opt-soo/45m-checkpoint/streaming_train_state_1000' \
    --mesh_dim='1,1,1' \
    --dtype='fp32' \
    --total_steps=27694 \
    --log_freq=122 \
    --eval_freq=122 \
    --eval_steps=100 \
    --save_model_freq=122 \
    --save_milestone_freq=0 \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='45M' \
    --llama.initializer_range=1.0 \
    --load_dataset_state='' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=122 \
    --optimizer.adamw_optimizer.lr_sched=constant_with_warmup \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr='0.01&0.003&0.03' \
    --optimizer.adamw_optimizer.init_lr=0.0 \
    --optimizer.adamw_optimizer.end_lr=0.0 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=227 \
    --optimizer.adamw_optimizer.b1=0.9 \
    --optimizer.adamw_optimizer.b2=0.95 \
    --weight_average=True \
    --weight_average_decay=0.9 \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.pretokenized_dataset_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/tokenized' \
    --train_dataset.huggingface_dataset.tokens_count_at_start=32768000 \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=32 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.pretokenized_dataset_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/tokenized-val' \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --checkpointer.save_optimizer_state=True \
    --wandb_project='EasyLM--opt-second-order-llama-45m' \
    --wandb_dir='' \
    --output_dir='' \
    --notes='Adam 45m bsz=4m' \
