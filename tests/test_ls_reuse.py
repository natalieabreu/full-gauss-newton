"""Exercise the trainer's batch routing on CPU, without importing TPU dependencies."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'


def trainer_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {'reused_ls_batch', 'validate_ls_reuse', 'pull_ls_batches_and_baseline'}
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    # Bind the nested function's enclosing state in the test namespace instead.
    class BindState(ast.NodeTransformer):
        def visit_Nonlocal(self, node):
            return ast.copy_location(ast.Global(names=node.names), node)
    module = BindState().visit(ast.Module(body=functions, type_ignores=[]))
    ns = {'np': np}
    exec(compile(module, str(SOURCE), 'exec'), ns)
    return ns


def batch(step):
    return {'input_tokens': np.full((2, 3), step, dtype=np.int32),
            'target_tokens': np.full((2, 3), step + 1, dtype=np.int32),
            'loss_masks': np.ones((2, 3), dtype=np.float32)}


class ReuseTests(unittest.TestCase):
    def test_startup_lag_and_host_copy(self):
        select = trainer_functions()['reused_ls_batch']
        for lag in (1, 3, 100):
            history = deque(maxlen=lag)
            for step in range(2 * lag + 3):
                current = batch(step)
                selected, age = select(history, current)
                expected = step - lag if step >= lag else max(0, step - 1)
                self.assertEqual(age, step - expected)
                np.testing.assert_array_equal(selected['input_tokens'], batch(expected)['input_tokens'])
                for key in current:
                    self.assertIsInstance(history[-1][key], np.ndarray)
                    self.assertFalse(np.shares_memory(current[key], history[-1][key]))
                current['input_tokens'][:] = -1
                self.assertEqual(history[-1]['input_tokens'][0, 0], step)
                self.assertLessEqual(len(history), lag)

    def test_actual_routing_and_fresh_baseline(self):
        ns = trainer_functions()
        pulled, logged, evaluated = [], [], []
        def pull(role):
            pulled.append(role)
            return batch(999), {}
        def loss(params, data, rng, micro):
            evaluated.append((params, data, rng))
            return float(np.mean((params - data['input_tokens']) ** 2)), None
        ns.update(FLAGS=NS(ls_reuse_lag=100, ls_eval_batches=1, inner_loop_iter=50, n_micro=1),
                  ls_history=deque(maxlen=100), ls_reused_tokens=0,
                  pull_training_batch=pull, defer_wandb=logged.append,
                  microbatched_loss_fn=loss,
                  jax=NS(random=NS(split=lambda rng: (rng + 1, rng + 2)), device_get=lambda x: x))
        run = ns['pull_ls_batches_and_baseline']
        for step in range(202):
            ns.update(step=step, single_batch_=batch(step))
            batches, rngs, _, baseline, exhausted = run(step, step + 0.5, None)
            source_step = step - 100 if step >= 100 else max(0, step - 1)
            self.assertEqual(baseline, (step + 0.5 - source_step) ** 2)
            self.assertIs(evaluated[-1][1], batches[0])
            self.assertEqual(evaluated[-1][2], rngs[0])
            self.assertFalse(exhausted)
            self.assertEqual(logged[-1]['ls_batch_step'], source_step)
            self.assertEqual(logged[-1]['phase_linesearch_reused_tokens'], (step + 1) * 6)
        self.assertEqual(pulled, [])  # No fresh read or fresh-token charge for LS.
        ns['FLAGS'].ls_reuse_lag = 0
        ns['FLAGS'].ls_eval_batches = 2
        batches, _, _, baseline, exhausted = run(0, 1000, None)
        self.assertEqual(pulled, ['linesearch', 'linesearch'])
        self.assertEqual(len(batches), 2)
        self.assertEqual(baseline, 1)
        self.assertEqual(ns['ls_reused_tokens'], 202 * 6)

    def test_unsupported_modes_fail_before_training(self):
        validate = trainer_functions()['validate_ls_reuse']
        def config():
            return NS(ls_reuse_lag=100, optimizer_type='adamw', gauss_newton=True,
                      single_batch_inner=True, adaptive_inner_loop=False, linesearch=True,
                      ls_eval_batches=1, train_batch_growth_interval=0,
                      train_dataset=NS(type='huggingface', huggingface_dataset=NS(pretokenized_dataset_dir='')),
                      load_dataset_state='', cg_resume_state='', save_model_freq=0, save_milestone_freq=0)
        validate(config())
        validate(NS(ls_reuse_lag=0))  # Legacy mode has no new restrictions.
        for key, value in [('ls_reuse_lag', -1), ('optimizer_type', 'cg'),
                           ('gauss_newton', False), ('single_batch_inner', False),
                           ('adaptive_inner_loop', True), ('linesearch', False),
                           ('ls_eval_batches', 2), ('train_batch_growth_interval', 1),
                           ('load_dataset_state', 'state'), ('cg_resume_state', 'state'),
                           ('save_model_freq', 1), ('save_milestone_freq', 1)]:
            flags = config()
            setattr(flags, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(flags)
        for kind, directory in [('json', ''), ('huggingface', '/tokens')]:
            flags = config()
            flags.train_dataset.type = kind
            flags.train_dataset.huggingface_dataset.pretokenized_dataset_dir = directory
            with self.assertRaises(ValueError):
                validate(flags)


if __name__ == '__main__':
    unittest.main()
