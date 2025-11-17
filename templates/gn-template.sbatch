#! /bin/bash

# This is the example script to pretrain a 45M LLaMA model with Gauss Newton.

#SBATCH -J gn-45m
#SBATCH -o printouts/gn-45m/%A_%a.out
#SBATCH -e printouts/gn-45m/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=1
#SBATCH -p 
#SBATCH --account=
#SBATCH --constraint="h100"
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=370G
#SBATCH -c 24


export PYTHONPATH="${PWD}:$PYTHONPATH"
export JAX_TRACEBACK_FILTERING=off
export WANDB_API_KEY=


module load python/3.10.12-fasrc01;
mamba activate EasyLM;


# TPU specific flags to improve training throughput
# export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python sweep_launcher.py \
    --program='EasyLM.models.llama.llama_train_gn' \
    --mesh_dim='1,1,1' \
    --load_checkpoint='trainstate_params::45m-checkpoint/streaming_train_state_1000' \
    --dtype='fp32' \
    --total_steps=227 \
    --log_freq=1 \
    --eval_freq=10 \
    --eval_steps=100 \
    --inner_loop_iter=122 \
    --gradient_accumulation_steps=1 \
    --save_model_freq=1 \
    --save_milestone_freq=0 \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='45M' \
    --llama.initializer_range=1.0 \
    --load_dataset_state='' \
    --optimizer_type='muon' \
    --lr_sched='constant_with_inner_cosine' \
    --inner_loop_lr=0.01 \
    --global_warmup=0 \
    --inner_loop_warmup=0 \
    --inner_loop_wd=0 \
    --parameter_wd=0 \
    --optimizer_wd=0.001 \
    --inner_b1=0.9 \
    --inner_b2=0.999 \
    --inner_clip_gradient=1 \
    --weight_average=True \
    --weight_average_decay=0.999 \
    --linesearch=True \
    --ls_range=5 \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.pretokenized_dataset_dir=${DATA_DIR}/train' \
    --train_dataset.huggingface_dataset.tokens_count_at_start=32768000 \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=32 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.pretokenized_dataset_dir='${DATA_DIR}/val' \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --checkpointer.save_optimizer_state=True \
    --wandb_project='EasyLM--opt-second-order-llama-45m' \
    --wandb_dir='' \
    --output_dir='' \
    --notes='GN 45M' \
    --gauss_newton=True \
    --reset_start=False \
    
