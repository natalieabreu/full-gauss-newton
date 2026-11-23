#!/usr/bin/env python3
"""Capacity watchdog for the two additional v4 Spot targets; never runs training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

PROJECT = 'fast-second-order'
ZONE = 'us-central2-b'
TARGETS = {'3': 'gn-v4-32-spot-1', '4': 'gn-v4-32-spot-2'}
OWNER = 'gfn-spot-queue'


def gcloud(group, action, *args):
    return ['gcloud', 'compute', 'tpus', group, action, *args,
            '--project=' + PROJECT, '--zone=' + ZONE, '--quiet']


def call(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=180,
                            env=dict(os.environ, CLOUDSDK_CORE_DISABLE_PROMPTS='1'))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'gcloud failed')
    return result.stdout


def inventory(group):
    # A failed list is never interpreted as an absent resource.
    data = json.loads(call(gcloud(group, 'list', '--format=json')))
    if not isinstance(data, list):
        raise ValueError('Unexpected inventory response')
    return {x['name'].rsplit('/', 1)[-1]: x for x in data}


def validate_owned(resource, name):
    specs = resource.get('tpu', {}).get('nodeSpec', [])
    if len(specs) != 1:
        raise ValueError(name + ': expected exactly one node specification')
    spec = specs[0]
    node = spec.get('node', {})
    if (spec.get('nodeId') != name or
            spec.get('parent') not in {f'projects/{PROJECT}/locations/{ZONE}',
                                       f'projects/598447672993/locations/{ZONE}'} or
            node.get('acceleratorType') != 'v4-32' or
            node.get('labels', {}).get('managed-by') != OWNER or
            'spot' not in resource):
        raise ValueError(name + ': existing request is not owned by this Spot watchdog')


def create_command(name, args):
    return gcloud('queued-resources', 'create', name, '--node-id=' + name,
                  '--accelerator-type=v4-32', '--spot', '--async',
                  '--runtime-version=' + args.runtime_version,
                  '--service-account=' + args.service_account,
                  '--scopes=https://www.googleapis.com/auth/cloud-platform',
                  '--labels=managed-by=' + OWNER,
                  '--metadata-from-file=startup-script=' + str(Path(args.startup_script).resolve()))


def reconcile(target, requests, nodes, args):
    name = TARGETS[target]
    resource = requests.get(name)
    if resource is None:
        if name in nodes:
            raise ValueError(name + ': node exists without our queued request; inspect manually')
        command = create_command(name, args)
    else:
        validate_owned(resource, name)
        state = resource.get('state', {}).get('state', 'UNKNOWN')
        print(name + ': ' + state, flush=True)
        if state not in {'SUSPENDED', 'FAILED'}:
            # Includes pending/active/deleting and unknown future states. Never reset a wait.
            return
        if name in nodes:
            raise ValueError(name + ': terminal request still has a node; refusing deletion')
        # No --force: server must also confirm that there are no nodes to delete.
        command = gcloud('queued-resources', 'delete', name, '--async')
    print(('APPLY ' if args.apply else 'PREVIEW ') + shlex.join(command), flush=True)
    if args.apply:
        call(command)
    # Recreation follows only after a later successful inventory confirms absence.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets', nargs='+', choices=sorted(TARGETS), default=['3'])
    p.add_argument('--config', required=True, help='queue.json from capture_setup.py')
    p.add_argument('--apply', action='store_true', help='Allow creates and terminal request cleanup')
    p.add_argument('--watch', action='store_true', help='Poll every 300 seconds')
    a = p.parse_args()
    config = json.loads(Path(a.config).expanduser().read_text())
    if set(config) != {'runtime_version', 'service_account', 'startup_script'}:
        p.error('Invalid queue configuration keys')
    for key, value in config.items():
        if not isinstance(value, str):
            p.error('Configuration values must be strings')
        setattr(a, key, value)
    if not Path(a.startup_script).is_file():
        p.error('startup-script must be a prepared local file')
    if not a.runtime_version or '@' not in a.service_account:
        p.error('Provide the verified runtime and service-account email')
    # Fixed per-project/zone lock: two controllers cannot maintain overlapping targets here.
    state = Path.home() / '.local/state/gfn-spot'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'watchdog.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            requests, nodes = inventory('queued-resources'), inventory('tpu-vm')
            for target in dict.fromkeys(a.targets):
                reconcile(target, requests, nodes, a)
            if not a.watch:
                break
            time.sleep(300)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit('Watchdog stopped; no further actions: ' + str(error))
