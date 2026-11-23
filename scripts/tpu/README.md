# Spot capacity for targets 3 and 4

These helpers maintain **capacity**, not an experiment queue or training resume.
They never provision, stop, delete or update fixed targets 1 and 2. Keep using the
original checkout for those targets; use a separate Git worktree for this branch
because launchers require matching controller/worker branch, SHA and helper hashes.

| Target | Node and queued-resource name | Location | Type |
| --- | --- | --- | --- |
| 3 | gn-v4-32-spot-1 | fast-second-order / us-central2-b | v4-32 Spot |
| 4 | gn-v4-32-spot-2 | fast-second-order / us-central2-b | v4-32 Spot |

Verified user report on 2026-11-23: reference TPU1 uses tpu-ubuntu2204-base,
Python 3.10.12, JAX/jaxlib 0.6.2, libtpu 0.0.17, Flax 0.10.7, Optax 0.2.8,
and service account 598447672993-compute@developer.gserviceaccount.com.
Do not use the old scripts/setup_tpu_vm.sh pins for this recovery.

## One-time preparation on box248

From a clean, pushed checkout of this branch, capture TPU1's runtime and installed
training dependency closure. This is read-only on TPU1; it never initializes JAX,
installs packages or changes its checkout. Requires working gcloud SSH access.

```bash
python3 scripts/tpu/capture_setup.py --output-dir "$HOME/.config/gfn-spot/setup-20261123"
```

The snapshot stores exact package versions and a startup script outside Git.
No tokens, environment dump, service-account keys or W&B keys are captured.
The worker uses an isolated venv with access to the base image's system packages
and installs the captured pins with --no-deps, avoiding resolver-driven upgrades.
It checks every captured version and imports the training modules on CPU. Existing
package-metadata conflicts are logged, not silently repaired. This reproduces
versions, not an immutable image or wheel hashes; package availability, native
libraries and new runtime-image revisions still need a real-worker check.

The attached service account must already have permission to read secret
`wandb-api-key` and read the warmup checkpoint in `gfn-soo`; output/queue writes
need the corresponding bucket permissions. IAM is never automatically modified.
Startup fetches the secret at runtime, checks access as ga2740 and reports FAILED
if setup/import/access checks fail. No secret value goes into VM metadata.
The initial bootstrap clones the pinned branch/SHA into /full-gauss-newton on each
new worker. Rerunning startup refuses an existing checkout, protecting live work.

## Preview, then maintain target 3

```bash
python3 scripts/tpu/spot_queue.py \
  --config "$HOME/.config/gfn-spot/setup-20261123/queue.json" --targets 3

mkdir -p "$HOME/.local/state/gfn-spot"
nohup python3 -u scripts/tpu/spot_queue.py \
  --config "$HOME/.config/gfn-spot/setup-20261123/queue.json" \
  --targets 3 --apply --watch \
  > "$HOME/.local/state/gfn-spot/watchdog.log" 2>&1 &
echo $! > "$HOME/.local/state/gfn-spot/watchdog.pid"
tail -f "$HOME/.local/state/gfn-spot/watchdog.log"
```

Preview reads cloud inventory but performs no mutation. Watch polls every 300s,
preserves pending/active/unknown states, and only deletes owned FAILED/SUSPENDED
requests when no node exists. Delete never uses --force. A later poll recreates an
absent request. Ownership checks require the exact node, project/zone, Spot tier,
v4-32 type and managed-by label. There is a local flock against duplicate watchdogs;
use box248 as the sole controller. API/auth/timeout errors stop with a visible log,
never masquerade as missing capacity. Review failed request details before retrying
if failures recur. Startup failure is not a capacity failure: an ACTIVE request
with FAILED setup is left in place for inspection.

**box248 credentials can still expire or require interactive reauthentication.**
nohup survives terminal disconnect, not host shutdown or authentication expiration.
An attached service account on workers does not authenticate box248. No claim of
permanent unattended recovery is made. If this recurs, a persistent controller VM
with attached identity is a separate infrastructure choice; do not add JSON keys.

## Check startup on all four workers

```bash
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32-spot-1 \
  --project=fast-second-order --zone=us-central2-b --worker=all \
  --command='cat /tmp/gfn_setup_status; sudo tail -n 30 /var/log/gfn-setup.log'
bash scripts/tpu.sh 3 status
```

After every worker reports DONE, check distributed topology simultaneously on all
workers (only on this idle new target):

```bash
gcloud compute tpus tpu-vm ssh ga2740@gn-v4-32-spot-1 \
  --project=fast-second-order --zone=us-central2-b --worker=all \
  --command="bash -lc 'source /etc/profile.d/gfn_env.sh && timeout 180 python3 -c \"import jax; jax.distributed.initialize(); print(jax.process_index(), jax.process_count(), jax.device_count(), jax.local_device_count()); assert (jax.process_count(), jax.device_count(), jax.local_device_count()) == (4,16,4); jax.distributed.shutdown()\"'"
```

Expect process indices 0..3, each reporting 4 processes, 16 global devices and 4
local devices. This is a topology check, not a training test. No training is
launched by the watchdog. GN parameter warm-start is not exact state/data resume.
Preempted runs need explicit operator decisions and available checkpoints.

## Expand to both targets / stop

In the controller shell, identify the watchdog before stopping it:

```bash
ps -p "$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")" -o pid,args
```

After verifying it is this Python watchdog, stop that process:

```bash
kill -TERM "$(cat "$HOME/.local/state/gfn-spot/watchdog.pid")"
```

Stopping the watchdog leaves existing TPU VMs and pending requests running. It
does not cancel capacity charges. To maintain both, relaunch the same nohup command
with `--targets 3 4` after the old process exits. Do not delete a node while the
watchdog is running unless recreation is intended. Use target 4's exact name for
its startup/topology checks.

## Local validation

```bash
python3 -m unittest discover -s scripts/tests -v
bash -n scripts/tpu/setup_vm.sh
```

Tests mock gcloud and exercise pending/terminal states, orphan/foreign resource
refusal, authentication failure, nonmutating preview and payload quoting. Cloud
provisioning, package downloads, IAM, actual startup and TPU execution require
operator validation. The scripts do not claim these local tests cover them.

API references:
- https://docs.cloud.google.com/tpu/docs/queued-resources
- https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/queued-resources/create
- https://docs.cloud.google.com/sdk/gcloud/reference/compute/tpus/queued-resources/delete
