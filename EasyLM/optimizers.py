import os
import time
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple
from functools import partial
import re
import dataclasses
import random

from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
import optax

from EasyLM.jax_utils import float_to_dtype

from soap_jax import soap
from flax.traverse_util import flatten_dict, unflatten_dict


class OptimizerFactory(object):
    """ Configurable optax optimizer factory. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.accumulate_gradient_steps = 1
        config.type = 'adamw'
        config.palm_optimizer = PalmOptimizerFactory.get_default_config()
        config.adamw_optimizer = AdamWOptimizerFactory.get_default_config()
        config.soap_optimizer = SOAPOptimizerFactory.get_default_config()
        config.muon_optimizer = MuonOptimizerFactory.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)
        if config.type == 'palm':
            optimizer, optimizer_info = PalmOptimizerFactory.get_optimizer(
                config.palm_optimizer, weight_decay_mask
            )
        elif config.type == 'adamw':
            optimizer, optimizer_info = AdamWOptimizerFactory.get_optimizer(
                config.adamw_optimizer, weight_decay_mask
            )
        elif config.type == 'soap':
            # uses adam for 1d params and soap for 2d params
            optimizer, optimizer_info = SOAPOptimizerFactory.get_optimizer(
                config.soap_optimizer, weight_decay_mask
            )
        elif config.type == 'muon':
            optimizer, optimizer_info = MuonOptimizerFactory.get_optimizer(
                config.muon_optimizer, weight_decay_mask
            )
        else:
            raise ValueError(f'Unknown optimizer type: {config.type}')

        if config.accumulate_gradient_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer, config.accumulate_gradient_steps
            )

        return optimizer, optimizer_info
# NA: Added support for Muon optimizer
class MuonOptimizerFactory(object):
    """ Muon optimizer factory."""
    def __init__(self):
        raise NotImplementedError
    
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr_sched = 'cosine'
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.beta = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = False
        config.multiply_by_parameter_scale = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        if config.lr_sched == 'cosine':
            learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
                decay_steps=config.lr_decay_steps,
                end_value=config.end_lr,
            )
        elif config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps > 0:
            learning_rate_schedule = optax.warmup_constant_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
            )
        elif config.lr_sched == 'constant' or (config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps == 0):
            learning_rate_schedule = optax.constant_schedule(config.lr)
        else:
            raise ValueError(f'Unknown inner loop schedule: {config.inner_loop_sched}')
        
        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )

        if config.multiply_by_parameter_scale:
            raise NotImplementedError
        
        adamw_chain = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                mask=weight_decay_mask,
                mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
        )

        muon_chain = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.contrib.muon(
                learning_rate=learning_rate_schedule,
                adam_weight_decay=config.weight_decay,
                adam_b1=config.b1,
                adam_b2=config.b2,
                beta=config.beta,
                mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
        )

        transform_dict = {
            'adamw': adamw_chain,
            'muon':   muon_chain,
        }

        def create_param_selector(params):
            """
            Return a pytree (same structure as params) whose leaves are strings
            ('adamw' or 'sgd'), AND print out the name of each parameter and its assignment.
            """
            # 1) Flatten the nested param dict so we get name tuples -> arrays
            flat_params = flatten_dict(params, sep='.')

            # Define first and last layer parameter names
            first_layer_keys = ['params.transformer.wte.embedding']
            last_layer_keys = ['params.lm_head.kernel']

            # 2) Build the selector tree
            flat_selector = {}
            for name_tuple, param in flat_params.items():
                print(name_tuple)
                if name_tuple in first_layer_keys or name_tuple in last_layer_keys:
                    print(f"Assigning param '{name_tuple}' (shape={param.shape}) to ADAMW.")
                    flat_selector[name_tuple] = 'adamw'
                else:
                    print(f"Assigning param '{name_tuple}' (shape={param.shape}) to MUON.")
                    flat_selector[name_tuple] = 'muon'

            # 3) Unflatten back to the original param-tree structure
            selector_tree = unflatten_dict(flat_selector, sep='.')
            return selector_tree

    
        def param_selector(params):
            return create_param_selector(params)
        
        optimizer = optax.multi_transform(transform_dict, param_selector)

        return optimizer, optimizer_info

    
# NA: Added support for SOAP optimizer
class SOAPOptimizerFactory(object):
    """ SOAP optimizer factory."""

    def __init__(self):
        raise NotImplementedError
    
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr_sched = 'cosine'
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = False
        config.multiply_by_parameter_scale = False
        config.precondition_frequency = 10

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        if config.lr_sched == 'cosine':
            learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
                decay_steps=config.lr_decay_steps,
                end_value=config.end_lr,
            )
        elif config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps > 0:
            learning_rate_schedule = optax.warmup_constant_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
            )
        elif config.lr_sched == 'constant' or (config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps == 0):
            learning_rate_schedule = optax.constant_schedule(config.lr)
        
        else:
            raise ValueError(f'Unknown inner loop schedule: {config.inner_loop_sched}')
        
        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )

        if config.multiply_by_parameter_scale:
            raise NotImplementedError
        
        adamw_chain = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                mask=weight_decay_mask,
                mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
        )

        # sgd_chain = optax.chain(
        #     optax.clip_by_global_norm(config.clip_gradient),
        #     # Feel free to add momentum, nesterov, etc. if needed:
        #     optax.sgd(learning_rate=learning_rate_schedule, momentum=0.0, nesterov=False),
        # )
        soap_chain = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            soap(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                precondition_frequency=config.precondition_frequency,
            )
        )
        # Put them into a dict that `multi_transform` will accept:
        transform_dict = {
            'adamw': adamw_chain,
            'soap':   soap_chain,
        }

        # def param_selector(params):
        #     def select_fn(param):
        #         # E.g.: param.ndim == 1 => "adamw", else => "sgd"
        #         # Adjust this logic as needed for your use case.
        #         return 'adamw' if param.ndim == 1 else 'soap'
        #     return jax.tree_map(select_fn, params)

        def create_param_selector(params):
            """
            Return a pytree (same structure as params) whose leaves are strings
            ('adamw' or 'sgd'), AND print out the name of each parameter and its assignment.
            """
            # 1) Flatten the nested param dict so we get name tuples -> arrays
            flat_params = flatten_dict(params, sep='.')

            # 2) Build a new flat dict, storing the transform-key ('adamw'/'sgd') for each param
            flat_selector = {}
            for name, param in flat_params.items():
                # (Optional) your own rule: if it's 1D => 'adamw', else 'sgd'
                if param.ndim == 1:
                    print(f"Assigning param '{name}' (shape={param.shape}) to ADAMW.")
                    flat_selector[name] = 'adamw'
                else:
                    print(f"Assigning param '{name}' (shape={param.shape}) to SOAP.")
                    flat_selector[name] = 'soap'

            # 3) Unflatten back to the original param-tree structure
            selector_tree = unflatten_dict(flat_selector, sep='.')
            return selector_tree
    
        def param_selector(params):
            return create_param_selector(params)
        
        optimizer = optax.multi_transform(transform_dict, param_selector)

        return optimizer, optimizer_info

class PalmOptimizerFactory(object):
    """ PaLM optimizer factory. This optimizer implements the optimizer
        described in the PaLM paper: https://arxiv.org/abs/2204.02311
    """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr = 0.01
        config.lr_warmup_steps = 10000
        config.b1 = 0.9
        config.b2 = 0.99
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        def learning_rate_schedule(step):
            multiplier = config.lr / 0.01
            return multiplier / jnp.sqrt(jnp.maximum(step, config.lr_warmup_steps))

        def weight_decay_schedule(step):
            multiplier = config.weight_decay / 1e-4
            return -multiplier * jnp.square(learning_rate_schedule(step))

        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
            weight_decay_schedule=weight_decay_schedule,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=config.b1,
                decay_rate=config.b2,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
            optax_add_scheduled_weight_decay(
                weight_decay_schedule, weight_decay_mask
            )
        )
        return optimizer, optimizer_info


class AdamWOptimizerFactory(object):
    """ AdamW optimizer with cosine schedule. """

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.lr_sched = 'cosine'
        config.init_lr = 0.0
        config.end_lr = 0.001
        config.lr = 0.01
        config.lr_warmup_steps = 2000
        config.lr_decay_steps = 500000
        config.b1 = 0.9
        config.b2 = 0.95
        config.clip_gradient = 1.0
        config.weight_decay = 1e-4
        config.bf16_momentum = False
        config.multiply_by_parameter_scale = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_optimizer(cls, config, weight_decay_mask=None):
        config = cls.get_default_config(config)

        if config.lr_sched == 'cosine':
            learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
                decay_steps=config.lr_decay_steps,
                end_value=config.end_lr,
            )
        elif config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps > 0:
            learning_rate_schedule = optax.warmup_constant_schedule(
                init_value=config.init_lr,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup_steps,
            )
        elif config.lr_sched == 'constant' or (config.lr_sched == 'constant_with_warmup' and config.lr_warmup_steps == 0):
            learning_rate_schedule = optax.constant_schedule(config.lr)
        else:
            raise ValueError(f'Unknown inner loop schedule: {config.lr_sched}')
        
        optimizer_info = dict(
            learning_rate_schedule=learning_rate_schedule,
        )

        if config.multiply_by_parameter_scale:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optax.adafactor(
                    learning_rate=learning_rate_schedule,
                    multiply_by_parameter_scale=True,
                    momentum=config.b1,
                    decay_rate=config.b2,
                    factored=False,
                    clipping_threshold=None,
                    dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
                ),
                optax_add_scheduled_weight_decay(
                    lambda step: -learning_rate_schedule(step) * config.weight_decay,
                    weight_decay_mask
                )
            )
        else:
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.clip_gradient),
                optax.adamw(
                    learning_rate=learning_rate_schedule,
                    weight_decay=config.weight_decay,
                    b1=config.b1,
                    b2=config.b2,
                    mask=weight_decay_mask,
                    mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
                ),
            )

        return optimizer, optimizer_info


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jax.Array


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """ Apply weight decay with schedule. """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
