"""CPU-only capacity state-machine checks. No cloud commands executed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tpu' / (name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
q, startup = module('spot_queue'), module('prepare_startup')


class SpotQueueTests(unittest.TestCase):
    def setUp(self):
        self.name = q.TARGETS['3']
        self.args = SimpleNamespace(apply=True, runtime_version='test-runtime',
                                    service_account='test@example.invalid', startup_script='/tmp/startup.sh')
    def resource(self, state):
        return {'spot': {}, 'state': {'state': state}, 'tpu': {'nodeSpec': [{
            'nodeId': self.name, 'parent': f'projects/{q.PROJECT}/locations/{q.ZONE}',
            'node': {'acceleratorType': 'v4-32', 'labels': {'managed-by': q.OWNER}}}]}}
    def test_only_new_targets(self):
        self.assertEqual(set(q.TARGETS), {'3', '4'})
        self.assertNotIn('gn-v4-32', q.TARGETS.values())
        self.assertNotIn('gn-v4-32-2', q.TARGETS.values())
    @patch.object(q, 'call')
    def test_missing_creates_spot(self, call):
        q.reconcile('3', {}, {}, self.args)
        command = call.call_args.args[0]
        self.assertIn('--spot', command)
        self.assertIn('--node-id='+self.name, command)
        self.assertIn('--async', command)
    @patch.object(q, 'call')
    def test_preserve_pending_active_and_unknown(self, call):
        for state in ['WAITING_FOR_RESOURCES', 'ACCEPTED', 'PROVISIONING', 'ACTIVE',
                      'SUSPENDING', 'CREATING', 'DELETING', 'FUTURE_STATE']:
            q.reconcile('3', {self.name:self.resource(state)}, {}, self.args)
        call.assert_not_called()
    @patch.object(q, 'call')
    def test_terminal_cleanup_never_force_or_create_same_poll(self, call):
        for state in ['SUSPENDED', 'FAILED']:
            call.reset_mock()
            q.reconcile('3', {self.name:self.resource(state)}, {}, self.args)
            call.assert_called_once()
            command = call.call_args.args[0]
            self.assertIn('delete', command)
            self.assertNotIn('--force', command)
            self.assertNotIn('create', command)
    @patch.object(q, 'call')
    def test_terminal_with_node_refused(self, call):
        with self.assertRaises(ValueError):
            q.reconcile('3', {self.name:self.resource('SUSPENDED')}, {self.name:{}}, self.args)
        call.assert_not_called()
    @patch.object(q, 'call')
    def test_orphan_node_refused(self, call):
        with self.assertRaises(ValueError):
            q.reconcile('3', {}, {self.name:{}}, self.args)
        call.assert_not_called()
    @patch.object(q, 'call')
    def test_foreign_resource_refused(self, call):
        for mutation in ['spot', 'labels', 'nodeId', 'parent']:
            resource=self.resource('SUSPENDED')
            spec=resource['tpu']['nodeSpec'][0]
            if mutation=='spot': del resource['spot']
            elif mutation=='labels': spec['node']['labels']={}
            else: spec[mutation]='gn-v4-32'
            with self.assertRaises(ValueError):
                q.reconcile('3', {self.name:resource}, {}, self.args)
        call.assert_not_called()
    @patch.object(q, 'call')
    def test_preview_no_mutations(self, call):
        self.args.apply=False
        q.reconcile('3', {}, {}, self.args)
        q.reconcile('3', {self.name:self.resource('FAILED')}, {}, self.args)
        call.assert_not_called()
    @patch.object(q, 'call', side_effect=RuntimeError('reauth required'))
    def test_auth_failure_not_absence(self, call):
        with self.assertRaisesRegex(RuntimeError, 'reauth'):
            q.inventory('queued-resources')
    def test_unreviewed_requirements_rejected(self):
        for requirements in ['', 'jax', 'jax @ https://example.invalid/file', '-e /local/repo']:
            with self.assertRaises(ValueError):
                startup.render(requirements, 'main', 'a'*40, '3.10', 'wandb-api-key')
    def test_launcher_target_mapping(self):
        import sys
        sys.path.insert(0, str(ROOT))
        import tpu_control
        self.assertEqual(tpu_control.DEFAULTS['targets']['1'], 'gn-v4-32')
        self.assertEqual(tpu_control.DEFAULTS['targets']['2'], 'gn-v4-32-2')
        for target, name in q.TARGETS.items():
            self.assertEqual(tpu_control.DEFAULTS['targets'][target], name)

    def test_payload_quoting_and_syntax(self):
        import subprocess
        requirements='jax==0.4.34\njaxlib==0.4.34\nlibtpu==0.0.1\nflax==0.8.3\noptax==0.2.2\n'
        result=startup.render(requirements, 'codex/spot-tpu-recovery', 'a'*40, '3.10', "quote'$(false)")
        subprocess.run(['bash', '-n'], input=result, text=True, check=True)
        self.assertNotIn('WANDB_API_KEY=quote', result)


if __name__ == '__main__':
    unittest.main()
