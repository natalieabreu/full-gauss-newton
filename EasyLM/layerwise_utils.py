from __future__ import annotations
from typing import Sequence, Callable, Any

import jax
from flax import struct, core
import optax

from typing import Any, Dict, Union, Mapping, List
from flax import traverse_util
from flax.core import FrozenDict

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

class TrainStateLayerwise(struct.PyTreeNode):
    """Stores *one* set of params but *multiple* optimizers / states
    (e.g. one per layer) for Gauss–Newton or other layer-wise schemes.
    """

    # — bookkeeping —
    step: int | jax.Array

    # — model —
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # — optimiser stacks (now sequences) —
    # Each entry in `tx` is an optax.GradientTransformation (stateless functions),
    # so we keep `pytree_node=False`.  The *states*, however, contain arrays and
    # must remain part of the PyTree (`True`).
    tx: Sequence[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Sequence[optax.OptState] = struct.field(pytree_node=True)
    
    warmstart_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True, default=None)

    # ------------------------------------------------------------------
    # Convenience constructors / updaters
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        params: core.FrozenDict,
        tx: Sequence[optax.GradientTransformation],
        apply_fn: Callable,
    ) -> "TrainStateLayerwise":
        """Initialise each optimizer's state against the same params."""
        # opt_state = [t.init(params) for t in tx]
        opt_state = []
        for i, t in enumerate(tx):
            print(f"Initializing optimizer {i}")
            opt_state.append(t.init(params))
            
        return cls(
            step=jax.numpy.array(0, dtype=jax.numpy.int32),
            apply_fn=apply_fn,
            params=params,
            tx=tuple(tx),          # tuples are marginally safer than lists
            opt_state=tuple(opt_state),
            warmstart_params=params,  # store initial params for warm-starting
        )

    def apply_gradients(
        self,
        grads: core.FrozenDict[str, Any],
    ) -> "TrainStateLayerwise":
        """Apply layer-wise gradient updates."""
        # 1) For each (tx_i, state_i) pair, compute updates
        updates_and_new_states = [
            tx_i.update(grads, state_i, self.params)
            for tx_i, state_i in zip(self.tx, self.opt_state)
        ]
        # 2) Merge / sum the per-layer updates (simple average shown;
        #    feel free to customise aggregation rule)
        updates = optax.apply_updates(
            self.params,
            sum((u for u, _ in updates_and_new_states))  # simple reduce-sum
        )
        new_states = [ns for _, ns in updates_and_new_states]

        return self.replace(
            step=self.step + 1,
            params=updates,
            opt_state=tuple(new_states),
        )
        
class TrainStateLayerwise2(struct.PyTreeNode):
    """Stores *one* set of params but *multiple* optimizers / states
    (e.g. one per layer) for Gauss–Newton or other layer-wise schemes.
    """

    # — bookkeeping —
    step: int | jax.Array

    # — model —
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # — optimiser stacks (now sequences) —
    # Each entry in `tx` is an optax.GradientTransformation (stateless functions),
    # so we keep `pytree_node=False`.  The *states*, however, contain arrays and
    # must remain part of the PyTree (`True`).
    tx: Sequence[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Sequence[optax.OptState] = struct.field(pytree_node=True)
    
    warmstart_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True, default=None)

    # ------------------------------------------------------------------
    # Convenience constructors / updaters
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        params: core.FrozenDict,
        tx: Sequence[optax.GradientTransformation],
        apply_fn: Callable,
        layers: Sequence[Union[str, int]] = None,
    ) -> "TrainStateLayerwise":
        """Initialise each optimizer's state against the same params."""
        # opt_state = [t.init(params) for t in tx]
        opt_state = []
        for i, t in enumerate(tx):
            print(f"Initializing optimizer {i}")
            layer_params = get_layer_params(params, layers[i], num_hidden_layers=len(tx)-2, sep='.')
            opt_state.append(t.init(layer_params))
            
        return cls(
            step=jax.numpy.array(0, dtype=jax.numpy.int32),
            apply_fn=apply_fn,
            params=params,
            tx=tuple(tx),          # tuples are marginally safer than lists
            opt_state=tuple(opt_state),
            warmstart_params=params,  # store initial params for warm-starting
        )

    def apply_gradients(
        self,
        grads: core.FrozenDict[str, Any],
    ) -> "TrainStateLayerwise":
        """Apply layer-wise gradient updates."""
        # 1) For each (tx_i, state_i) pair, compute updates
        updates_and_new_states = [
            tx_i.update(grads, state_i, self.params)
            for tx_i, state_i in zip(self.tx, self.opt_state)
        ]
        # 2) Merge / sum the per-layer updates (simple average shown;
        #    feel free to customise aggregation rule)
        updates = optax.apply_updates(
            self.params,
            sum((u for u, _ in updates_and_new_states))  # simple reduce-sum
        )
        new_states = [ns for _, ns in updates_and_new_states]

        return self.replace(
            step=self.step + 1,
            params=updates,
            opt_state=tuple(new_states),
        )

class LayerState(struct.PyTreeNode):
    """Stores the parameters and optimizer state for a single layer."""
    # arrays/pytree leaves
    layer_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    step: jax.Array = struct.field(pytree_node=True)
    full_structure: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

    # static (non-traced) metadata
    layer: Union[int, str] = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    apply_fn: Callable = struct.field(pytree_node=False)
    num_hidden_layers: int = struct.field(pytree_node=False)
    sep: str = struct.field(pytree_node=False, default='.')

    @classmethod
    def create(
        cls,
        *,
        full_params: core.FrozenDict,
        tx: optax.GradientTransformation,
        apply_fn: Callable,
        layer: Union[int, str],
        num_hidden_layers: int,
        sep: str = '.',
    ) -> "LayerState":
        """Initialize optimizer state and capture correct layer slicing rules."""
        opt_state = tx.init(full_params)
        layer_params = get_layer_params(full_params, layer, num_hidden_layers=num_hidden_layers, sep=sep)
        return cls(
            layer_params=layer_params,
            opt_state=opt_state,
            step=jax.numpy.array(0, dtype=jax.numpy.int32),
            full_structure=jax.tree_util.tree_map(jnp.zeros_like, full_params),

            layer=layer,                   # static
            tx=tx,                         # static
            apply_fn=apply_fn,             # static
            num_hidden_layers=num_hidden_layers,  # static
            sep=sep,
        )

    def apply_gradients(self, grads: core.FrozenDict[str, Any]) -> "LayerState":
        """Apply gradient updates to the layer parameters, using full-tree optax."""
        # full grads tree
        full_grads = merge_layer_params(
            self.full_structure, self.layer, grads,
            num_hidden_layers=self.num_hidden_layers, sep=self.sep
        )
        # full current params tree (needed for optax.multi_transform masks)
        full_params = merge_layer_params(
            self.full_structure, self.layer, self.layer_params,
            num_hidden_layers=self.num_hidden_layers, sep=self.sep
        )

        updates, new_states = self.tx.update(full_grads, self.opt_state, params=full_params)

        updated_full = optax.apply_updates(full_params, updates)

        # extract the single layer back out
        layer_updates = get_layer_params(
            updated_full, self.layer,
            num_hidden_layers=self.num_hidden_layers, sep=self.sep
        )

        return self.replace(
            layer_params=layer_updates,
            opt_state=new_states,
            step=self.step + 1,
        )
    
def get_layer_params(
    params: core.FrozenDict,
    layer: Union[int, str],
    *,
    num_hidden_layers: int,
    sep: str = "."
) -> Dict[str, Any]:
    """
    Return a nested dict containing only the parameters of the given layer,
    keeping full key structure intact.
    """
    # Flatten the parameters
    if isinstance(params, FrozenDict):
        params = params.unfreeze()
    flat_params = traverse_util.flatten_dict(params, sep=sep)

    # Get valid keys for the layer
    layer_keys = get_layer_param_keys(params, layer, sep=sep, num_hidden_layers=num_hidden_layers)

    if not layer_keys:
        raise ValueError(f"No parameters found for layer {layer}")

    # Build filtered dict of only the layer's parameters
    layer_subtree = {k: flat_params[k] for k in layer_keys}

    # Re-nest and return
    nested_layer_params = traverse_util.unflatten_dict(layer_subtree, sep=sep)
    return nested_layer_params
    
def merge_layer_params(
    full_params: FrozenDict,
    layer: Union[int, str],
    layer_params: Mapping[str, Any],
    *,
    num_hidden_layers: int,
    sep: str = ".",
) -> FrozenDict:
    """
    Replace the parameters of a specific layer in `full_params` with `layer_params`.

    Assumes `layer_params` was generated by `get_layer_params(...)`,
    meaning it uses **full key paths** (e.g. 'params.transformer.h.3.kernel').

    Returns a new FrozenDict with the updated parameters.
    """
    # 1. Flatten everything
    if isinstance(full_params, FrozenDict):
        full_params = full_params.unfreeze()
    flat_full = traverse_util.flatten_dict(full_params, sep=sep)

    flat_layer = traverse_util.flatten_dict(layer_params, sep=sep)

    # 2. Get all keys belonging to the given layer
    layer_keys = get_layer_param_keys(full_params, layer, sep=sep, num_hidden_layers=num_hidden_layers)

    # 3. Remove existing keys for that layer
    for key in layer_keys:
        if key in flat_full:
            del flat_full[key]

    # 4. Insert new layer parameters (already prefixed)
    flat_full.update(flat_layer)

    # 5. Re-nest and freeze
    nested_full = traverse_util.unflatten_dict(flat_full, sep=sep)
    return nested_full


def print_layer_param_keys(params: FrozenDict, layer: Union[int, str], *, sep: str = ".") -> None:
    """
    Print all flattened parameter keys corresponding to a given layer.

    Args:
        params: FrozenDict of model parameters.
        layer:  Layer identifier — int (e.g. 3 → "transformer.h.3") or string (e.g. "wte").
        sep:    Separator for flattening dictionary keys (default: ".").
    """
    # Flatten the FrozenDict
    if isinstance(params, FrozenDict):
        params = params.unfreeze()
    flat_params = traverse_util.flatten_dict(params, sep=sep)

    # Build the key prefix for the selected layer
    if isinstance(layer, int):
        layer_key = f"params.transformer.h.{layer}"
    elif isinstance(layer, str) and layer != 'lm_head':
        layer_key = f"params.transformer.{layer}"
    elif layer == 'lm_head':
        layer_key = 'params.lm_head'
    else:
        raise TypeError(f"`layer` must be int or str, got {type(layer)}")

    # Print all keys that start with this layer's prefix
    print(f"** Parameter keys for layer '{layer_key}':")
    found = False
    for key in flat_params:
        if key.startswith(layer_key) or key.startswith(f'transformer.{layer_key}' + sep):
            print(f"  {key}")
            found = True

    if not found:
        print(f"  [No parameters found for layer '{layer_key}']")
        
def print_param_partition_assignments(params: Any, param_selector_fn, sep="."):
    """
    Print out the optimizer partition assigned to each parameter.

    Args:
        params: Nested parameter dict (FrozenDict or plain dict).
        param_selector_fn: Function mapping each parameter to a partition label.
        sep: Separator used when flattening keys.
    """
    # Get the per-parameter partition labels
    partitions = param_selector_fn(params)

    # Flatten both param and partition trees for aligned display
    flat_partitions = traverse_util.flatten_dict(partitions, sep=sep)
    flat_params = traverse_util.flatten_dict(params, sep=sep)

    print("Parameter → Optimizer Assignment")
    print("-" * 40)
    for k in sorted(flat_params.keys()):
        partition = flat_partitions.get(k, '[MISSING]')
        print(f"{k:60} → {partition}")
        
def _get_layer_param_keys(
    params: FrozenDict,
    layer: Union[int, str],
    *,
    sep: str = ".",
) -> List[str]:
    """
    Return all flattened parameter-dict keys that correspond to a specific layer.

    Args:
        params: A Flax `FrozenDict` (or nested dict) containing model parameters.
        layer:  The layer identifier:
                • int  → uses "transformer.h.{layer}"
                • str  → "wte", "ln_f", etc.  Special-cased "lm_head".
        sep:    Separator used when flattening (default ".").

    Returns:
        List[str]: All keys whose prefix matches the requested layer.
                   The list is empty if no parameters are found.
    """
    # 1) Flatten the parameter PyTree into { "params.xxx": ... } keys
    if isinstance(params, FrozenDict):
        params = params.unfreeze()
    flat_params = traverse_util.flatten_dict(params, sep=sep)  # keys are strings

    # 2) Build the canonical prefix for this layer
    if isinstance(layer, int):                              # transformer blocks
        prefix = f"params.transformer.h.{layer}"
    elif isinstance(layer, str) and layer != "lm_head":     # wte, ln_f, …
        prefix = f"params.transformer.{layer}"
    elif layer == "lm_head":                                # output head
        prefix = "params.lm_head"
    else:
        raise TypeError(f"`layer` must be int or str, got {type(layer)}")

    # 3) Collect keys that start with the prefix
    keys = [k for k in flat_params if k.startswith(prefix)]

    # 4) (Optional) fall-back: some checkpoints omit the leading "params."
    if not keys:
        alt_prefix = prefix.replace("params.", "", 1)
        keys = [k for k in flat_params if k.startswith(alt_prefix)]

    return keys

def get_layer_param_keys(
    params: FrozenDict,
    layer: int,
    *,
    num_hidden_layers: int,
    sep: str = "."
) -> List[str]:
    """
    Return keys for a transformer block + handles ln_f grouping with last transformer layer
    """
    keys = _get_layer_param_keys(params, layer, sep=sep)

    if isinstance(layer, int) and layer == num_hidden_layers - 1:
        keys += _get_layer_param_keys(params, "ln_f", sep=sep)

    return keys

def cosine_similarity(params_a, params_b):
    # flatten both pytrees to 1-D vectors
    vec_a, _ = ravel_pytree(params_a)
    vec_b, _ = ravel_pytree(params_b)

    # standard cosine-sim
    dot   = jnp.vdot(vec_a, vec_b)              # inner product
    norma = jnp.linalg.norm(vec_a)
    normb = jnp.linalg.norm(vec_b)

    return dot / (norma * normb + 1e-8)         # small eps to avoid /0

def norm_ratio(params_a, params_b):
    vec_a, _ = ravel_pytree(params_a)
    vec_b, _ = ravel_pytree(params_b)

    norm_a = jnp.linalg.norm(vec_a)
    norm_b = jnp.linalg.norm(vec_b)

    return norm_b / (norm_a + 1e-8)  # avoid divide-by-zero

def norm(params_a):
    vec_a, _ = ravel_pytree(params_a)

    norm_a = jnp.linalg.norm(vec_a)

    return norm_a  # avoid divide-by-zero