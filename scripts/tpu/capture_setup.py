#!/usr/bin/env python3
"""Read TPU1's installed dependency closure and prepare new Spot workers locally."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
from prepare_startup import render

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = r'''
import importlib.metadata as md
import json, sys
from packaging.requirements import Requirement
roots = ['jax','jaxlib','libtpu','flax','optax','transformers','tokenizers','datasets',
         'mlxu','einops','gcsfs','google-cloud-storage','wandb','tqdm','psutil']
seen, pending, pins = set(), [(x, ()) for x in roots], {}
while pending:
    name, extras = pending.pop()
    key = (name.lower().replace('_','-'), tuple(sorted(extras)))
    if key in seen: continue
    seen.add(key)
    d = md.distribution(name)
    pins[d.metadata['Name']] = d.version
    for raw in d.requires or []:
        req = Requirement(raw)
        if req.marker is None or any(req.marker.evaluate({'extra': e}) for e in ('', *extras)):
            pending.append((req.name, tuple(req.extras)))
print('GFN_SNAPSHOT=' + json.dumps({'python_minor': '%d.%d' % sys.version_info[:2],
      'requirements': '\n'.join(k+'=='+v for k,v in sorted(pins.items()))+'\n'}))
'''


def run(argv):
    return subprocess.check_output(argv, text=True, timeout=180)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--wandb-secret', default='wandb-api-key')
    a = p.parse_args()
    directory = Path(a.output_dir).expanduser().resolve()
    if directory.exists():
        p.error('Use a new output directory; existing snapshots are preserved')
    branch = run(['git', '-C', str(ROOT), 'branch', '--show-current']).strip()
    sha = run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip()
    if not branch or run(['git', '-C', str(ROOT), 'status', '--porcelain']).strip():
        p.error('Use a clean, pushed named branch before capturing setup')
    flags = ['--project=fast-second-order', '--zone=us-central2-b']
    info = json.loads(run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe', 'gn-v4-32',
                          *flags, '--format=json']))
    if info.get('acceleratorType') != 'v4-32':
        p.error('Reference TPU is not v4-32')
    if info.get('networkConfig', {}).get('network', '').rsplit('/',1)[-1] != 'default':
        p.error('Reference uses a custom network; configure networking explicitly before provisioning')
    code = base64.b64encode(CAPTURE.encode()).decode()
    command = 'source /etc/profile.d/gfn_env.sh && python3 -c ' + shlex.quote(
        "import base64; exec(base64.b64decode(" + repr(code) + '))')
    output = run(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', 'ga2740@gn-v4-32',
                  *flags, '--worker=0', '--command='+shlex.join(['bash','-lc',command])])
    records = [line.removeprefix('GFN_SNAPSHOT=') for line in output.splitlines()
               if line.startswith('GFN_SNAPSHOT=')]
    if len(records) != 1:
        raise ValueError('Expected exactly one dependency snapshot')
    snapshot = json.loads(records[0])
    body = render(snapshot['requirements'], branch, sha, snapshot['python_minor'], a.wandb_secret)
    directory.mkdir(parents=True)
    (directory/'requirements.txt').write_text(snapshot['requirements'])
    (directory/'startup.sh').write_text(body)
    config = dict(runtime_version=info['runtimeVersion'],
                  service_account=info['serviceAccount']['email'],
                  startup_script=str(directory/'startup.sh'))
    (directory/'queue.json').write_text(json.dumps(config, indent=2)+'\n')
    print('Prepared '+str(directory)+' at '+branch+' '+sha)
    print('Capture reads the reference worker; it does not install or launch anything.')


if __name__ == '__main__':
    main()
