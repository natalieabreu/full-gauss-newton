import pprint
from functools import partial

from google.cloud import storage

from tqdm import tqdm, trange
import numpy as np
import mlxu
import subprocess as sp
import neural_tangents as nt

import timeit
import os
import wandb
import copy

import jax
import jax.numpy as jnp
from jax import linearize, linear_transpose
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer
from flax.traverse_util import flatten_dict, unflatten_dict

import optax

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint, cross_entropy_loss_and_accuracy_with_weight_decay, CustomTrainState
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLMModule
)
from EasyLM.gcs_utils import (
    load_ckpt_from_gcs, load_from_gcs, 
    upload_to_gcs, load_first_n_files_from_gcs, 
    modify_dataset_info_gcs, modify_state_json_gcs
)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    log_inner_steps=False,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_freq=0,
    eval_steps=0,
    gradient_accumulation_steps=1,
    inner_loop_iter=100,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset_batch_size=8,
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    # optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    # logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    outer_loop_method='replace',
    lr_sched='cosine',
    inner_loop_lr=0.001,
    inner_loop_wd=0.0,
    end_lr=0.0,
    global_warmup=0.2,
    inner_loop_warmup=0.0,

    optimizer_type='adamw',
    inner_b1=0.9,
    inner_b2=0.999,
    inner_clip_gradient=0.0,
    optimizer_wd=0.0,
    parameter_wd=0.0,

    wandb_run_id='',
    start_tokens=0,

    wandb_project='',
    wandb_entity='',
    wandb_dir='SOO-LM/experiment_output/open_llama_7b',
    output_dir='',
    notes='',
    logger=mlxu.WandBLogger.get_default_config(),
    experiment_id='',
    
    # GCS specific flags
    gcs_num_train_files_to_download=300,
    tmp_dir='/tmp',

    weight_average=False,
    weight_average_decay=0.99,
    load_ema_checkpoint='',
    linesearch=False,
    ls_range=5,

    gauss_newton=False,
    redo_gn=0,
    reset_start=False,

    target_loss=0.0
)

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def is_embedding_param(param_name, param_value):
    if 'embedding' in param_name:
        return True
    return False

def count_params(params):
    non_embedding_count = 0
    total_count = 0

    for param_name, param_value in jax.tree_util.tree_leaves_with_path(params):
        # print(param_name[-1].key, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        total_count += jnp.prod(jnp.array(param_value.size))
        if not is_embedding_param(param_name[-1].key, param_value):
            non_embedding_count += jnp.prod(jnp.array(param_value.size))
            print(param_name[-5:], is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        else:
            print(param_name, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
    # print(non_embedding_count)
    return total_count, non_embedding_count



def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_id)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    log_config = mlxu.flatten_config_dict(flags_config_dict)

    set_random_seed(FLAGS.seed)

    print(FLAGS.train_dataset)
    init_checkpoint_path = FLAGS.load_checkpoint

    if FLAGS.load_checkpoint.split('::')[-1].startswith('gs://'):
        FLAGS.load_checkpoint = load_ckpt_from_gcs(FLAGS.load_checkpoint, local_path=os.path.join(FLAGS.tmp_dir, 'model.ckpt'))
    if FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir.startswith('gs://'):
        num_to_download = FLAGS.gcs_num_train_files_to_download # Files download around 100 MiB/s
        tmp_dir = FLAGS.tmp_dir
        load_first_n_files_from_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train'), os.path.join(tmp_dir, 'train_dataset/train'), num_to_download=num_to_download)
        modify_dataset_info_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train/dataset_info.json'), os.path.join(tmp_dir, 'train_dataset/train'), num_files_to_keep=num_to_download)
        modify_state_json_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train/state.json'), os.path.join(tmp_dir, 'train_dataset/train'), num_files_to_keep=num_to_download)
        load_from_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'dataset_dict.json'), os.path.join(tmp_dir, 'train_dataset/dataset_dict.json'))
        FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir = os.path.join(tmp_dir, 'train_dataset')
    if FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir.startswith('gs://'):
        FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir = load_from_gcs(FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir, os.path.join(FLAGS.tmp_dir,'eval_dataset'))
    if FLAGS.load_dataset_state != '' and FLAGS.load_dataset_state.startswith('gs://'):
        FLAGS.load_dataset_state = load_from_gcs(FLAGS.load_dataset_state, os.path.join(FLAGS.tmp_dir, 'dataset_state.pkl')) 

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))
        print('loaded dataset state', flush=True)

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    def get_global_lr_sched(method, lr, taylor_steps, inner_loop_iter, warmup, inner_warmup, end_lr):
        if method == 'global_cosine':
            decay_steps = taylor_steps*inner_loop_iter
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)

            if isinstance(warmup, tuple):
                warmup = int(warmup[0])
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
        elif method == 'cosine_with_global_schedule':
            decay_steps = taylor_steps
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)
            if isinstance(warmup, tuple):
                warmup = int(warmup[0])

            if inner_warmup <= 1.0:
                inner_warmup = int(inner_warmup*inner_loop_iter)
            if isinstance(inner_warmup, tuple):
                inner_warmup = int(inner_warmup[0])
            
            global_sched = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            schedules = []
            boundaries = []
            for step in range(taylor_steps):
                peak_lr = global_sched(step)
                inner_sched = optax.warmup_cosine_decay_schedule(
                    init_value=peak_lr*0.1,
                    peak_value=peak_lr,
                    warmup_steps=inner_warmup,
                    decay_steps=inner_loop_iter,
                    end_value=end_lr,
                )
                schedules.append(inner_sched)
                boundaries.append(step*inner_loop_iter)

            schedule = optax.join_schedules(schedules, boundaries)

        elif method == 'constant_with_inner_cosine':
            decay_steps = taylor_steps
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)
            if isinstance(warmup, tuple):
                warmup = int(warmup[0])

            if inner_warmup <= 1.0:
                inner_warmup = int(inner_warmup*inner_loop_iter)
            if isinstance(inner_warmup, tuple):
                inner_warmup = int(inner_warmup[0])

            if warmup == 0:
                init_value = lr
            else:
                init_value = lr*0.1
            
            global_sched = optax.warmup_constant_schedule(
                init_value=init_value,
                peak_value=lr,
                warmup_steps=warmup,
            )
            schedules = []
            boundaries = []
            for step in range(taylor_steps):
                peak_lr = global_sched(step)
                inner_sched = optax.warmup_cosine_decay_schedule(
                    init_value=peak_lr*0.1,
                    peak_value=peak_lr,
                    warmup_steps=inner_warmup,
                    decay_steps=inner_loop_iter,
                    end_value=end_lr,
                )
                schedules.append(inner_sched)
                boundaries.append((step+1)*inner_loop_iter)

            schedule = optax.join_schedules(schedules, boundaries[:-1])

        elif method == 'constant':
            schedule = optax.constant_schedule(lr)
        else:
            raise ValueError(f"Unknown global schedule method: {method}")

        return schedule

    def build_optimizer(lr_sched, b1, b2, grad_clip=None, wd=0.0, optimizer_type='adamw'):
        if optimizer_type == 'adamw':
            if grad_clip:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(grad_clip),
                    optax.adamw(
                        learning_rate=lr_sched,
                        b1=b1,
                        b2=b2,
                        mu_dtype=jnp.float32,
                        weight_decay=wd
                    )
                )
            else:
                optimizer = optax.adamw(
                    learning_rate=lr_sched,
                    b1=b1,
                    b2=b2,
                    mu_dtype=jnp.float32,
                    weight_decay=wd
                )
        elif optimizer_type == 'muon':
            adamw_chain = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(
                    learning_rate=lr_sched,
                    weight_decay=wd,
                    b1=b1,
                    b2=b2,
                    mu_dtype=jnp.float32,
                ),
            )

            muon_chain = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.contrib.muon(
                    learning_rate=lr_sched,
                    adam_weight_decay=wd,
                    adam_b1=b1,
                    adam_b2=b2,
                    mu_dtype=jnp.float32,
                ),
            )

            transform_dict = {
                'adamw': adamw_chain,
                'muon':   muon_chain,
            }

            def create_param_selector(params):
                """
                Return a pytree (same structure as params) whose leaves are strings
                ('adamw' or 'muon'), AND print out the name of each parameter and its assignment.
                """
                # 1) Flatten the nested param dict so we get name tuples -> arrays
                flat_params = flatten_dict(params, sep='.')

                # Define first and last layer parameter names
                first_layer_keys = ['params.transformer.wte.embedding']
                last_layer_keys = ['params.lm_head.kernel']

                # 2) Build the selector tree
                flat_selector = {}
                for name_tuple, param in flat_params.items():
                    # print(name_tuple)
                    if name_tuple in first_layer_keys or name_tuple in last_layer_keys:
                        # print(f"Assigning param '{name_tuple}' (shape={param.shape}) to ADAMW.")
                        flat_selector[name_tuple] = 'adamw'
                    else:
                        # print(f"Assigning param '{name_tuple}' (shape={param.shape}) to MUON.")
                        flat_selector[name_tuple] = 'muon'

                # 3) Unflatten back to the original param-tree structure
                selector_tree = unflatten_dict(flat_selector, sep='.')
                return selector_tree

        
            def param_selector(params):
                return create_param_selector(params)
            
            optimizer = optax.multi_transform(transform_dict, param_selector)
        return optimizer

    # optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)
    lr_sched = get_global_lr_sched(FLAGS.lr_sched, FLAGS.inner_loop_lr, FLAGS.total_steps, FLAGS.inner_loop_iter, FLAGS.global_warmup, FLAGS.inner_loop_warmup, FLAGS.end_lr)
    tayl_solver = build_optimizer(lr_sched, FLAGS.inner_b1, FLAGS.inner_b2, FLAGS.inner_clip_gradient, FLAGS.optimizer_wd, FLAGS.optimizer_type)

    # optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    def create_trainstate_from_params(params):
        return CustomTrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return CustomTrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def train_step_jvp(train_state, params0, rng, batch, wd):
        rng_generator = JaxRNG(rng)

        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params0, params):
            dparams = jax.tree_util.tree_map(lambda x, y: x - y, params, params0)
            def f_batch(p):
                logits = model.apply(
                    p, batch['input_tokens'], deterministic=False,
                    rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                ).logits
                return logits
            primals, Jx = jax.jvp(f_batch, (params0,), (dparams,))
            logits = primals + jax.lax.stop_gradient(Jx)
            return cross_entropy_loss_and_accuracy_with_weight_decay(
                logits, batch['target_tokens'], train_state.params, params0, batch['loss_masks'], weight_decay=wd
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(params0, train_state.params)
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            learning_rate=lr_sched(train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics


    def train_step_gauss_newton(train_state, params0, rng, batch, wd):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def f_batch(p):
            out = model.apply(
                p,
                batch['input_tokens'],
                deterministic=False,              
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            )
            return out.logits                    # [B, ..., vocab]

        def scalar_loss_on_logits(logits):
            loss, _ = cross_entropy_loss_and_accuracy_with_weight_decay(
                logits, batch['target_tokens'], train_state.params, params0, batch['loss_masks'], weight_decay=wd
            )
            return loss

        def value_and_gradient(params0, params):
            '''
            ∇θ [ L(y0) + g0·v + 1/2 v^T G0 v ]
            = g0 + H0 v
                g0 = ∂L/∂p at p0 = ∂L/∂f @ ∂f/∂p at p0 ;  
            H0 v = (∂²L/∂p² at p0) @ v = (g0^T ∂²L/∂f² g0) (dθ) = (J(p0)^T ∂²L/∂f² J(p0) (dθ))
            '''
            # Linearize f at params0
            logits0, jvp_fn = linearize(f_batch, params0)          # y0,   v = J(p0) dθ

            # dθ and forward-mode JVP: v = J0 (params - params0)
            dparams = jax.tree_util.tree_map(lambda x, y: x - y, params, params0)
            v = jvp_fn(dparams)

            # g0 = ∂L/∂y at y0 ;  Hv = (∂²L/∂y² at y0) @ v
            grad_Ly = jax.grad(scalar_loss_on_logits)              # y -> grad wrt logits
            g0 = grad_Ly(logits0) # ∂L/∂f at p0
            _, Hv = jax.jvp(grad_Ly, (logits0,), (v,))            # Hessian-vector (logits space) = (∂²L/∂f² at p0) J(p0) dθ

            # Single pullback: J0^T (g0 + H0 v)
            jt_fn = linear_transpose(jvp_fn, params0) # primals just for shape/dtype
            (grad_params,) = jt_fn(jax.tree_util.tree_map(lambda a, b: a + b, g0, Hv))

            # quadratic loss on linear model
            loss = scalar_loss_on_logits(logits0) + jnp.sum(g0 * v) + 0.5 * jnp.sum(v * Hv)


            return (loss, 0), grad_params

        (loss, accuracy), grads = value_and_gradient(params0, train_state.params)

        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")

        train_state = train_state.apply_gradients(grads=grads)

        metrics = dict(
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            learning_rate=lr_sched(train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics


    def loss_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)

        logits = model.apply(
            params, batch['input_tokens'], deterministic=False,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        return cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )


    def eval_step(params, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
            eval_perplexity=perplexity,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    batch_partition = {
        'input_tokens': PS(('dp', 'fsdp')), 
        'loss_masks': PS(('dp', 'fsdp')),
        'target_tokens': PS(('dp', 'fsdp')),
    }

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.gauss_newton:
        sharded_train_step = pjit(
            train_step_gauss_newton,
            in_shardings=(train_state_partition, train_state_partition.params, PS(), batch_partition, PS()),
            out_shardings=(train_state_partition, PS(), PS()),
            # donate_argnums=(0, 1),
        )
    else:

        sharded_train_step = pjit(
            train_step_jvp,
            in_shardings=(train_state_partition, train_state_partition.params, PS(), batch_partition, PS()),
            out_shardings=(train_state_partition, PS(), PS()),
            # donate_argnums=(0, 1),
        )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition.params, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, ema=None, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            ema=ema,
            # dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
    

    def shard_batch(batch, num_devices):
        # Shard each tensor along the first axis
        sharded = {k: np.array_split(v, num_devices) for k, v in batch.items()}
        # Group the shards for each device into a list of dictionaries
        return [{k: sharded[k][i] for k in batch} for i in range(num_devices)]

    


    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    print(f"Mesh axes names: {mesh.axis_names}")
    print(f"Mesh shape: {mesh.shape}")

    with mesh:
        print(mesh)
        train_state, restored_params = None, None
        warmstart_params = None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            # distinguish between loading from train_state and loading from params
            if train_state is not None and output_dir in init_checkpoint_path: # need to distinguish between loading adam initial ckpt and taylor mid-run ckpt
                # dataset_path = os.path.join(output_dir, 'dataset.pkl')
                # dataset.load_state_dict(mlxu.load_pickle(dataset_path))
                
                if FLAGS.weight_average:
                    _, ema = checkpointer.load_trainstate_checkpoint(
                        FLAGS.load_ema_checkpoint, train_state_shapes, shard_fns
                    )

                if FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir != '':
                    start_step = int(jax.device_get(train_state.step))
                    start_tokens = int(jax.device_get(train_state.step)) * FLAGS.train_dataset_batch_size * seq_length + FLAGS.train_dataset.huggingface_dataset.tokens_count_at_start
                    dataset.set_start_tokens(start_tokens)
                    print('loaded checkpoint, starting at step', start_step, flush=True)
                    print('\tstart tokens:', start_tokens)

            if train_state is not None: # do this in both cases
                opt_state = train_state.opt_state
                if train_state.warmstart_params:
                    warmstart_params = train_state.warmstart_params

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        # param_count = sum(x.size for x in jax.tree_leaves(train_state.params))
        param_count, param_count_nonembed = count_params(train_state.params)
        param_count = jax.device_get(param_count)
        param_count_nonembed = jax.device_get(param_count_nonembed)

        flags_config_dict['param_count'] = param_count
        flags_config_dict['param_count_nonembed'] = param_count_nonembed

        if FLAGS.wandb_run_id:
            wandb.init(entity=FLAGS.wandb_entity, project=FLAGS.wandb_project, resume="must", id=FLAGS.wandb_run_id, dir=FLAGS.wandb_dir)
        else:
            wandb.init(entity=FLAGS.wandb_entity, project=FLAGS.wandb_project, config=log_config, dir=FLAGS.wandb_dir)  # Replace with your project name

            is_gcs = output_dir.startswith("gs://")

            # If not GCS, create local directory
            if not is_gcs and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save wandb_id.txt locally first
            local_path = os.path.join(output_dir if not is_gcs else FLAGS.tmp_dir, "wandb_id.txt")

            with open(local_path, 'w+') as f:
                f.write(wandb.run.id)  # Hacky but easier than handling in train state loader

            # If output_dir is a GCS bucket, upload the file
            if is_gcs:
                gcs_path = os.path.join(output_dir, "wandb_id.txt")
                upload_to_gcs(local_path, gcs_path)


        start_step = int(jax.device_get(train_state.step))
        
        def copy_array(x):
            return copy.copy(x)  # or x.copy() if x is a NumPy/JAX array

        if FLAGS.save_model_freq > 0:
            if FLAGS.weight_average:
                ema = jax.tree.map(copy_array, train_state.params)
                save_checkpoint(train_state, ema=ema)
            else:
                save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        assert FLAGS.train_dataset_batch_size % mesh.shape['dp'] == 0, \
            "Batch size must be divisible by the number of devices in 'dp'."
        
        
        
        if FLAGS.weight_average:
            print('Using weight average')
            ema = jax.tree.map(copy_array, train_state.params)


        inner_state = create_trainstate_from_params(train_state.params)
        dataset = iter(dataset)

        if warmstart_params is not None and not FLAGS.reset_start:
            print('Using warmstart params')
            inner_state = inner_state.replace(params=warmstart_params)

        for step in step_counter:
            print("step", step, "param norm", global_norm(train_state.params), flush=True)

            if FLAGS.reset_start:
                inner_state = inner_state.replace(
                    params=train_state.params,
                    opt_state=tayl_solver.init(train_state.params)
                )

            for i in range(FLAGS.inner_loop_iter):
                batch_, dataset_metrics_ = next(dataset)
                batch = jax.tree.map(
                    lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                    batch_
                )
                inner_state, sharded_rng, metrics = sharded_train_step(
                    inner_state, train_state.params, sharded_rng, batch, FLAGS.inner_loop_wd
                )

                if FLAGS.log_inner_steps:
                    log_metrics = {"inner_step": step*FLAGS.inner_loop_iter + i}
                    log_metrics['inner_loss'] = metrics['loss']
                    log_metrics['inner_gradient_norm'] = metrics['gradient_norm']
                    log_metrics['inner_param_norm'] = metrics['param_norm']
                    log_metrics['inner_gpu_memory'] = metrics['gpu_memory']
                    log_metrics['inner_learning_rate'] = metrics['learning_rate']
                    wandb.log(log_metrics)

                if FLAGS.weight_average and not FLAGS.linesearch: # for linesearch, do ema in outer loop
                    alpha = FLAGS.weight_average_decay
                    ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, inner_state.params)

            # train_state = train_state.replace(params=jax.device_get(inner_state.params), step=step)

            if FLAGS.linesearch:
                    
                exit = False
                pre_fetched_batches = []
                for _ in range(FLAGS.inner_loop_iter): # new data
                    try:
                        batch, _ = next(dataset)
                        pre_fetched_batches.append(batch)
                    except StopIteration:
                        print('Dataset exhausted')
                        exit = True
                        break
                
                if exit:
                    break
                # loss_partial = partial(compute_average_loss, dataset=dataset, rng=sharded_rng, loss_fn=loss_fn, batch_accumulation_steps=FLAGS.inner_loop_iter*gradient_accumulation_steps)
                dir = jax.tree_util.tree_map(lambda x, y: x - y, inner_state.params, train_state.params)
                losses = []
                for step_size in [1/jnp.sqrt(2)**i for i in range(FLAGS.ls_range)]:
                    # Compute loss using pre-fetched batches
                    updated_params = jax.tree_util.tree_map(lambda x, y: x + step_size*y, train_state.params, dir)
                    accumulated_loss = 0.0
                    for batch in pre_fetched_batches:
                        sharded_rng, subrng = jax.random.split(sharded_rng)
                        loss, _ = loss_fn(updated_params, batch, subrng)
                        accumulated_loss += loss
                    
                    average_loss = accumulated_loss / len(pre_fetched_batches)
                    losses.append((step_size, average_loss))
                step_size, loss = min(losses, key=lambda x: x[1])
                step_size = jax.device_get(step_size)
                print('Step size:', step_size)
                wandb.log({
                    'step_size': step_size,
                    'global_step': step,
                    })
                # for (_step_size, _loss) in losses:
                #     tag = f"{_step_size:.4f}"    
                #     wandb.log({
                #         f"step_size_{tag}_loss": _loss,
                #         "global_step": global_step,
                #     })
                updated_params = jax.tree_util.tree_map(lambda x, y: x + step_size*y, train_state.params, dir)
                train_state = train_state.replace(
                    step=train_state.step+1,
                    opt_state=inner_state.opt_state,
                    params=updated_params,
                    warmstart_params=inner_state.params, # store the new params for the next iteration
                )
                
                if FLAGS.weight_average:
                    alpha = FLAGS.weight_average_decay
                    ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, updated_params)
            else:
                train_state = train_state.replace(
                    step=train_state.step+1,
                    opt_state=inner_state.opt_state,
                    params=inner_state.params
                )
       
            if step % FLAGS.log_freq == 0:
                log_metrics = {"global_step": step}
                log_metrics.update(metrics)
                # log_metrics.update(dataset_metrics)
                

                do_eval = FLAGS.eval_freq and FLAGS.eval_steps > 0 and ((step % FLAGS.eval_freq == 0 and step <= FLAGS.total_steps * 0.5) or (step % FLAGS.log_freq == 0 and step > FLAGS.total_steps * 0.5))

                if do_eval: # eval_freq must be | by log_freq
                    eval_iterator = iter(eval_dataset)
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)

                        if FLAGS.weight_average:
                            eval_params=ema
                        else:
                            eval_params = train_state.params
                        sharded_rng, eval_metrics = sharded_eval_step(
                            eval_params, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    log_metrics.update(average_metrics(eval_metric_list))
                    
                    if FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] <= FLAGS.target_loss:
                        print(f"Target loss {FLAGS.target_loss} reached with loss {log_metrics['eval_loss']}, stopping at step {step}")
                        log_metrics = jax.device_get(log_metrics)
                        wandb.log(log_metrics)
                        tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
                        
                        break
                    elif FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] >= 15:
                        print(f"Loss {log_metrics['eval_loss']} too high, stopping at step {step}")
                        break
                    # metrics.update({"step": step})
                    # metrics = jax.device_get(metrics)
                    # logger.log(metrics)
                log_metrics = jax.device_get(log_metrics)
                wandb.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
            
            

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                if FLAGS.weight_average:
                    ema = jax.device_get(ema)
                    save_checkpoint(train_state, ema=ema, milestone=True)
                else:
                    save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                if FLAGS.weight_average:
                    ema = jax.device_get(ema)
                    save_checkpoint(train_state, ema=ema)
                else:
                    save_checkpoint(train_state)

        if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
            eval_iterator = iter(eval_dataset)
            eval_metric_list = []
            for _ in range(FLAGS.eval_steps):
                eval_batch, _ = next(eval_iterator)

                if FLAGS.weight_average:
                    eval_params=ema
                else:
                    eval_params = train_state.params
                sharded_rng, eval_metrics = sharded_eval_step(
                    eval_params, sharded_rng, eval_batch
                )
                eval_metric_list.append(eval_metrics)
            log_metrics.update(average_metrics(eval_metric_list))
            log_metrics = jax.device_get(log_metrics)
            wandb.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

    wandb.finish()


if __name__ == "__main__":
    print(jax.local_devices())
    print(jax.devices())
    mlxu.run(main)
