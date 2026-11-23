#!/usr/bin/env bash
# Startup payload body; prepend settings with prepare_startup.py. Runs as root.
set -euo pipefail
umask 022
exec >> /var/log/gfn-setup.log 2>&1
trap 'echo "FAILED $(date -Is)" > /tmp/gfn_setup_status' ERR
echo "STARTING $(date -Is)" > /tmp/gfn_setup_status
: "${GFN_BRANCH:?}" "${GFN_SHA:?}" "${GFN_REQUIREMENTS_B64:?}" "${GFN_PYTHON_MINOR:?}"
: "${GFN_WANDB_SECRET:?}"
# An existing checkout may have active training: startup reruns never change it.
if [[ -e /full-gauss-newton ]]; then
  echo 'Existing checkout: refusing automatic reconfiguration.'
  false
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git tmux python3-venv
actual_python=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ "$actual_python" == "$GFN_PYTHON_MINOR" ]]
id ga2740 >/dev/null 2>&1 || useradd --create-home --shell /bin/bash ga2740
python3 -m venv --system-site-packages /opt/gfn-venv
printf '%s' "$GFN_REQUIREMENTS_B64" | base64 --decode > /opt/gfn-requirements.txt
/opt/gfn-venv/bin/python -m pip install --no-deps -r /opt/gfn-requirements.txt \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Reproduce observed versions, including any pre-existing dependency metadata conflicts.
# A resolver must not silently upgrade the working JAX stack.
/opt/gfn-venv/bin/python -m pip check || echo 'Dependency metadata conflicts recorded above.'
/opt/gfn-venv/bin/python - <<'PYVERIFY'
import importlib.metadata as md
from pathlib import Path
for line in Path('/opt/gfn-requirements.txt').read_text().splitlines():
    name, expected = line.split('==')
    assert md.version(name) == expected, (name, md.version(name), expected)
PYVERIFY
git clone --branch "$GFN_BRANCH" https://github.com/gmbabreu/full-gauss-newton.git /full-gauss-newton
cd /full-gauss-newton
git cat-file -e "$GFN_SHA^{commit}"
git checkout -B "$GFN_BRANCH" "$GFN_SHA"
chown -R ga2740:ga2740 /full-gauss-newton
# Fetch the W&B secret with attached service-account credentials; never store it in Git/metadata.
command -v gcloud
{
  printf 'source /opt/gfn-venv/bin/activate\n'
  printf 'export PYTHONPATH=/full-gauss-newton${PYTHONPATH:+:$PYTHONPATH}\n'
  printf 'export WANDB_PROJECT=gfn-soo\n'
  printf 'export WANDB_ENTITY=ga2740-new-york-university\n'
  printf 'WANDB_API_KEY=$(gcloud secrets versions access latest --project=fast-second-order --secret=%q) || return 1\n' "$GFN_WANDB_SECRET"
  printf 'export WANDB_API_KEY\n'
} > /etc/profile.d/gfn_env.sh
chmod 644 /etc/profile.d/gfn_env.sh
# Import check only: avoid initializing a distributed TPU backend independently on each worker.
runuser -u ga2740 -- bash -c 'source /etc/profile.d/gfn_env.sh && test -n "$WANDB_API_KEY" && JAX_PLATFORMS=cpu python3 -c "import EasyLM.models.llama.llama_model, EasyLM.data, EasyLM.optimizers; from libtpu.sdk import monitoring; from google.cloud import storage; assert storage.Client().bucket(\"gfn-soo\").blob(\"checkpoints/warmup_150m/1_0/streaming_train_state\").exists()"'
echo "DONE $(date -Is) $GFN_SHA; TPU topology and access still require operator verification" > /tmp/gfn_setup_status
