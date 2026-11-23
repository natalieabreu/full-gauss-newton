#!/usr/bin/env python3
"""Build a secret-free startup script from a reviewed exact package snapshot."""
import argparse
import base64
from pathlib import Path
import re
import shlex
import subprocess


def render(requirements, branch, sha, python_minor, secret):
    lines = [s.strip() for s in requirements.splitlines() if s.strip() and not s.startswith('#')]
    if not lines or any(not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', s) for s in lines):
        raise ValueError('Requirements must contain only reviewed package==version pins')
    names = {s.split('==')[0].lower().replace('_', '-') for s in lines}
    if not {'jax', 'jaxlib', 'flax', 'optax', 'libtpu'} <= names and not {
            'jax', 'jaxlib', 'flax', 'optax', 'libtpu-nightly'} <= names:
        raise ValueError('Snapshot must include JAX, jaxlib, libtpu (or libtpu-nightly), Flax and Optax')
    if not re.fullmatch('[0-9a-f]{40}', sha) or not re.fullmatch(r'3\.\d+', python_minor):
        raise ValueError('Need a full commit SHA and Python major.minor')
    subprocess.run(['git', 'check-ref-format', '--branch', branch], check=True, capture_output=True)
    values = dict(GFN_BRANCH=branch, GFN_SHA=sha, GFN_PYTHON_MINOR=python_minor,
                  GFN_WANDB_SECRET=secret,
                  GFN_REQUIREMENTS_B64=base64.b64encode(('\n'.join(lines)+'\n').encode()).decode())
    body = Path(__file__).with_name('setup_vm.sh').read_text()
    payload = '#!/usr/bin/env bash\n' + ''.join(k+'='+shlex.quote(v)+'\n' for k,v in values.items()) + body
    if len(payload.encode()) > 250000:
        raise ValueError('Startup payload is too large for inline metadata')
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requirements', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--python-minor', required=True)
    p.add_argument('--wandb-secret', default='wandb-api-key')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = render(Path(a.requirements).read_text(), a.branch, a.sha, a.python_minor, a.wandb_secret)
    with Path(a.output).open('x') as output:
        output.write(result)


if __name__ == '__main__':
    main()
