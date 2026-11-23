#!/usr/bin/env python3
"""Controller CLI. No trainer, cloud SDK, or third-party Python imports."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

from run_experiment import ROOT, choices, make_plan, preview
from tpu_worker import encode, fingerprint

# All site defaults and optional environment overrides live here.
DEFAULTS = dict(
    targets={'1': 'gn-v4-32', '2': 'gn-v4-32-2',
             '3': 'gn-v4-32-spot-1', '4': 'gn-v4-32-spot-2'},
    project=os.environ.get('GFN_PROJECT', 'fast-second-order'),
    zone=os.environ.get('GFN_ZONE', 'us-central2-b'),
    user=os.environ.get('GFN_SSH_USER', 'ga2740'),
    repo=os.environ.get('GFN_REMOTE_REPO', '/full-gauss-newton'),
    environment=os.environ.get('GFN_REMOTE_ENV', '/etc/profile.d/gfn_env.sh'),
    state=os.environ.get('GFN_STATE_DIR', '/tmp/gfn-tpu-ga2740'),
)
HELP = '''Usage (from root: bash scripts/tpu.sh; from scripts/: bash tpu.sh):
  tpu.sh TARGET run PRESET [--dry-run] [--flag=value ...]
  tpu.sh TARGET status
  tpu.sh TARGET logs [WORKER]       Follow latest log; default worker 0
  tpu.sh TARGET stop                Stop managed runs only
  tpu.sh TARGET pull                Fast-forward the current branch
  tpu.sh TARGET switch BRANCH       Fetch/switch/fast-forward on all workers
  tpu.sh TARGET ssh [WORKER]        Interactive shell; default worker 0
  tpu.sh help
TARGET: 1 = gn-v4-32; 2 = gn-v4-32-2;
        3 = gn-v4-32-spot-1; 4 = gn-v4-32-spot-2
Presets (editable starter settings): ''' + ', '.join(choices()) + '''
gn_sgd selects the inner GN solver. adam selects ordinary AdamW training.
See docs/tpu_workflow.md for sync, checkpoint and replay restrictions.
'''


def local_git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


class Controller:
    def __init__(self, target):
        self.cfg = dict(DEFAULTS)
        self.cfg['target'] = os.environ.get('GFN_TPU_NAME', DEFAULTS['targets'][target])
        self.token = uuid.uuid4().hex
        self.workers = []
        self.reserved = []
        self.base = dict(repo=self.cfg['repo'], environment=self.cfg['environment'],
                         state=self.cfg['state'], target=self.cfg['target'], token=self.token,
                         fingerprint=fingerprint(ROOT))

    def gcloud(self, *args):
        return ['gcloud', 'compute', 'tpus', 'tpu-vm', *args,
                '--project=' + self.cfg['project'], '--zone=' + self.cfg['zone']]

    def discover(self):
        result = subprocess.run(self.gcloud('describe', self.cfg['target'], '--format=json'),
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Cannot discover workers: ' + result.stderr.strip())
        endpoints = json.loads(result.stdout).get('networkEndpoints', [])
        if not endpoints:
            raise RuntimeError('TPU description has no networkEndpoints')
        self.workers = list(range(len(endpoints)))

    def ssh(self, worker, remote=None, *, stream=False):
        args = self.gcloud('ssh', self.cfg['user'] + '@' + self.cfg['target'], '--worker=' + str(worker))
        if remote is not None:
            args += ['--command=' + remote]
        return subprocess.run(args, text=True, capture_output=not stream)

    def rpc(self, worker, action, extra=None, *, stream=False):
        payload = dict(self.base, worker=worker, **(extra or {}))
        # Shell is used solely to source the site's environment. Every dynamic
        # argument is quoted; JSON is transported as one base64 argv element.
        setup = ('set -eo pipefail; '
                 + 'if [ -f ' + shlex.quote(self.cfg['environment']) + ' ]; then source '
                 + shlex.quote(self.cfg['environment']) + '; fi; cd -- ' + shlex.quote(self.cfg['repo'])
                 + '; exec ' + shlex.join(['python3', 'scripts/tpu_worker.py', action, encode(payload)]))
        result = self.ssh(worker, shlex.join(['bash', '-lc', setup]), stream=stream)
        if stream:
            if result.returncode:
                raise RuntimeError(f'worker {worker}: {action} ended with status {result.returncode}')
            return {}
        lines = [s[len('GFN_RESULT='):] for s in result.stdout.splitlines() if s.startswith('GFN_RESULT=')]
        if not lines:
            raise RuntimeError(f'worker {worker}: {action}: no worker response; '
                               + (result.stderr.strip() or result.stdout.strip()))
        response = json.loads(lines[-1])
        if not response['ok'] or result.returncode:
            raise RuntimeError(f'worker {worker}: {action}: ' + response.get('error', result.stderr.strip()))
        return response['result']

    def all_workers(self, action, extra=None):
        values, errors = {}, []
        for worker in self.workers:
            try:
                values[worker] = self.rpc(worker, action, extra)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))
        return values

    def reserve(self):
        errors = []
        for worker in self.workers:
            # Record even ambiguous failures: acquire may have succeeded before
            # a dropped SSH acknowledgement. Release can only remove our token.
            self.reserved.append(worker)
            try:
                self.rpc(worker, 'acquire')
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))

    def release(self):
        errors = []
        for worker in self.reserved:
            try:
                self.rpc(worker, 'release')
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('Reservation cleanup incomplete; inspect workers:\n' + '\n'.join(errors))

    def run(self, plan):
        dirty = local_git('status', '--porcelain', '--untracked-files=all')
        if dirty:
            raise RuntimeError('Controller working tree is dirty; commit tooling and presets before launching')
        self.base.update(expected_sha=local_git('rev-parse', 'HEAD'),
                         expected_branch=local_git('branch', '--show-current'))
        if not self.base['expected_branch']:
            raise RuntimeError('Controller must be on a named branch')
        self.all_workers('preflight')
        print('All workers passed preflight. Starting ' + plan['run_id'], flush=True)
        try:
            for worker in self.workers:
                result = self.rpc(worker, 'launch', {'plan': plan})
                print(f'worker {worker}: {result["message"]}; {result["log"]}', flush=True)
        except BaseException as error:
            errors = []
            # Clean only this launch, including ambiguous SSH outcomes.
            for worker in self.workers:
                try:
                    self.rpc(worker, 'stop', {'stop_run_id': plan['run_id']})
                except Exception as cleanup:
                    errors.append(str(cleanup))
            raise RuntimeError(f'Partial launch failure: {error}\n'
                               f'Cleanup attempted for run {plan["run_id"]} on workers {self.workers}. '
                               + ('Cleanup incomplete: ' + '; '.join(errors) if errors else 'Owned processes stopped.')) from error
        print('Processes started on all workers. Use status/logs to check training initialization.', flush=True)

    def update(self, branch=None):
        before = self.all_workers('preflight_update')
        identities = {(r['branch'], r['sha']) for r in before.values()}
        if len(identities) != 1:
            raise RuntimeError('Workers differ in branch/SHA; reconcile explicitly before updating')
        current_branch, _ = next(iter(identities))
        branch = branch or current_branch
        if not branch or branch.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]*', branch):
            raise ValueError('Invalid or detached target branch')
        targets = self.all_workers('fetch', {'branch': branch})
        shas = {r['target_sha'] for r in targets.values()}
        if len(shas) != 1:
            raise RuntimeError('Workers fetched different commits; nothing switched. Rerun once origin is stable')
        target = next(iter(shas))
        try:
            self.all_workers('update', {'branch': branch, 'target_sha': target})
            after = self.all_workers('status')
            if any(r['sha'] != target or r['branch'] != branch or r['dirty'] for r in after.values()):
                raise RuntimeError('Workers did not converge cleanly')
        except Exception as error:
            raise RuntimeError(f'Update incomplete; workers may differ. Target {branch} {target}. '
                               'Inspect status; no rollback/reset was attempted.\n' + str(error)) from error
        print(f'All workers converged: {branch} {target}')

    def operate(self, action, plan=None, branch=None):
        self.discover()
        # Local flock plus token reservations on every remote worker serialize
        # managed launches/updates/stops across controllers.
        key = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()[:20]
        lock_path = Path('/tmp') / ('gfn-controller-' + key + '.lock')
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another controller operation is active for this target')
            try:
                self.reserve()
                if action == 'run':
                    self.run(plan)
                elif action in ('pull', 'switch'):
                    self.update(branch)
                elif action == 'stop':
                    results = self.all_workers('stop')
                    for worker, report in results.items():
                        print(f'worker {worker}: ' + json.dumps(report))
                        if report['unmanaged'] or report['remaining_tmux']:
                            print(f'worker {worker}: unmanaged/legacy activity remains; inspect it explicitly')
            except BaseException as error:
                try:
                    self.release()
                except Exception as cleanup:
                    raise RuntimeError(str(error) + '\n' + str(cleanup)) from error
                raise
            else:
                self.release()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args in ([], ['help'], ['--help'], ['-h']):
        print(HELP)
        return 0
    if len(args) < 2 or args[0] not in DEFAULTS['targets']:
        raise ValueError(HELP)
    target, action, *rest = args
    if action not in {'run', 'status', 'logs', 'stop', 'pull', 'switch', 'ssh'}:
        raise ValueError(HELP)
    controller = Controller(target)
    if action == 'run':
        if not rest or rest[0].startswith('--'):
            raise ValueError('run requires a preset. Choose: ' + ', '.join(choices()))
        preset, *overrides = rest
        dry = '--dry-run' in overrides
        overrides = [arg for arg in overrides if arg != '--dry-run']
        plan = make_plan(preset, overrides)
        print(f'Target: {controller.cfg["target"]} ({controller.cfg["project"]}, {controller.cfg["zone"]})\n'
              f'Preset: {preset}; run ID: {plan["run_id"]}\n'
              f'Output: {plan["output_location"]}\nCommand: {preview(plan)}\n'
              'Launch plan: discover workers; reserve ALL; preflight clean/matching branch/SHA, tooling, '
              'environment, tmux and processes on ALL; start managed tmux/process groups; '
              'W&B enabled only on worker 0; record logs/manifests outside Git.\n'
              f'Route: {plan["route"]}; PYTHONUNBUFFERED=1; '
              + (f'SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_JOB_ID={plan["run_id"]}; launcher appends _0.\n'
                 if plan['route'] == 'sweep' else 'direct module execution.\n'), flush=True)
        if dry:
            print('DRY RUN: no external commands, imports of trainers, or launches performed.')
            return 0
        controller.operate(action, plan)
    elif action in {'logs', 'ssh'}:
        if len(rest) > 1 or (rest and not rest[0].isdigit()):
            raise ValueError(action + ' accepts one nonnegative worker number')
        worker = int(rest[0]) if rest else 0
        controller.discover()
        if worker not in controller.workers:
            raise ValueError(f'Worker must be in {controller.workers}')
        if action == 'ssh':
            return controller.ssh(worker, stream=True).returncode
        controller.rpc(worker, 'logs', stream=True)
    elif action == 'status':
        if rest:
            raise ValueError('status accepts no additional arguments')
        controller.discover()
        reports, errors = {}, []
        for worker in controller.workers:
            try:
                reports[worker] = controller.rpc(worker, 'status')
                print(f'worker {worker}: ' + json.dumps(reports[worker], indent=2))
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('\n'.join(errors))
    else:
        if (action == 'switch' and len(rest) != 1) or (action != 'switch' and rest):
            raise ValueError('switch requires one branch; stop/pull accept no extra arguments')
        controller.operate(action, branch=rest[0] if rest else None)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
