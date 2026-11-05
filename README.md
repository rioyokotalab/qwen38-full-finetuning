# Qwen3.8-27B full-parameter fine-tuning

Portable Slurm recipe for full-parameter supervised fine-tuning of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Every model
parameter is trainable: this repository does **not** use LoRA/PEFT,
quantization, or frozen modules.

The scripts are intended for a multi-node NVIDIA cluster with Slurm, a shared
filesystem, and RDMA-capable interconnect. The machine-specific values live in
one environment file; no RIKYU account, host name, or path is embedded in the
code.

## What has been demonstrated

The original configuration completed a one-step systems acceptance run on six
nodes / 24 NVIDIA GB200 GPUs (192 GiB per GPU): NCCL all-reduce across all 24
ranks, loading the exact model revision below, FSDP2 full sharding, all
27,356,728,560 parameters trainable, and finite training loss. That run used
PyTorch 2.12.1+cu130 from a site wheelhouse. The public default in this
repository is PyTorch 2.9.1+cu130, so a new cluster must pass the included smoke
test before a research run. The prior result is evidence about the recipe, not
a performance or compatibility guarantee for other hardware.

Generic activation checkpointing is deliberately disabled. In the validated
stack it changed tensor metadata while recomputing Qwen3.8's hybrid
DeltaNet/attention blocks and PyTorch rejected the backward pass. Do not turn
it on without a separately tested Qwen-specific implementation.

## Hardware and storage

- Linux cluster with Slurm and one Slurm task per node
- NVIDIA GPUs with BF16 support; fast RDMA/NVLink-class communication is
  strongly recommended
- Shared project, dataset, model-cache, log, and checkpoint paths visible at
  identical paths on every node
- Rough sizing: BF16 weights alone are about 55 GB. Gradients, optimizer state,
  FSDP all-gathers, activations, and CUDA workspaces make the practical
  requirement much larger. Start qualification with 16–24 GPUs of at least
  80 GB each; the accepted reference used 24 x 192 GB. The smoke test—not this
  estimate—is the admission gate for a particular cluster.
- Reserve at least 1 TB of shared storage for model cache, a sharded optimizer
  checkpoint, logs, and working headroom.

## 1. Clone and install

The following commands are run on a login node. `uv` is the only bootstrap
dependency; it creates `.venv` and installs the locked Python environment.

```bash
git clone git@github.com:rioyokotalab/qwen38-full-finetuning.git
cd qwen38-full-finetuning
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is installed
uv sync --locked
cp config/cluster.env.example config/cluster.env
```

The lock targets Python 3.12, Linux x86-64, CUDA 13.0, and PyTorch's official
`cu130` wheel index. For aarch64/SBSA or a site-provided PyTorch build, keep the
application pins but regenerate the lock only after choosing and testing that
site's official torch source. Never mix CUDA wheel families in one environment.

## 2. Configure the cluster

Edit `config/cluster.env`:

```bash
PROJECT_ROOT=/shared/$USER/qwen38-full-finetuning
SLURM_PARTITION=gpu
SLURM_ACCOUNT=my_project              # may be blank
NODES=6
GPUS_PER_NODE=4
GPU_REQUEST_STYLE=per-node            # per-node, total, or gres
CPUS_PER_TASK=32
TIME_LIMIT=01:00:00
HF_HOME=/shared/$USER/cache/huggingface
```

`GPU_REQUEST_STYLE` controls the scheduler request only:

- `per-node`: `--gpus-per-node=N`
- `total`: `--gpus=NODES*GPUS_PER_NODE`
- `gres`: `--gres=gpu:N`

Copy or clone this repository to `PROJECT_ROOT`, and ensure `.venv` and all
data/output paths are visible on every compute node. If modules are required,
put their `module load ...` commands in `config/site_setup.sh`; the supplied
file is intentionally empty.

## 3. Download and verify the model

The commit is pinned in `model.lock`, so later upstream changes cannot silently
alter an experiment.

```bash
set -a; source config/cluster.env; source model.lock; set +a
mkdir -p "$HF_HOME"
HF_HOME="$HF_HOME" uv run hf download "$MODEL_ID" --revision "$MODEL_REVISION"
```

Access to the model is governed by its
[model card and license](https://huggingface.co/Qwen/Qwen3.8-27B). Model weights
are not stored in this repository.

## 4. Run the acceptance smoke test

```bash
scripts/submit_smoke.sh config/cluster.env
```

This submits one Slurm job that first proves NCCL collectives and exact rank /
node / GPU coverage, then runs one full-parameter training step on
`data/smoke.jsonl`. Do not start a real experiment unless the log contains all
three markers:

```text
NCCL_ACCEPTED
FULL_PARAMETER_ACCEPTED
TRAINING_STEP_ACCEPTED
```

The smoke file is a systems fixture, not research data.

## 5. Prepare the dataset

Input is UTF-8 JSONL. Each non-empty line must contain a non-empty `messages`
array in Qwen chat-template format:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

The current loader is text-only. All vision parameters remain trainable and in
the optimizer, but a claim that the vision encoder was adapted requires an
image-bearing loader and actual image examples.

Before a research run, record the immutable dataset manifest (path, row count,
SHA-256, provenance, license, and train/evaluation split) outside the dataset
itself. Do not commit private data or credentials.

## 6. Run an experiment

Use absolute shared paths:

```bash
export TRAIN_DATA=/shared/project/data/train.jsonl
export OUTPUT_DIR=/shared/project/outputs/qwen38-run-001
export MAX_LENGTH=4096
export MAX_STEPS=1000
export GRADIENT_ACCUMULATION_STEPS=4
export LEARNING_RATE=1e-6
export SAVE_CHECKPOINT=1
scripts/submit_train.sh config/cluster.env
```

The submit script prints the accepted Slurm job ID. Monitor it with the normal
site tools (`squeue`, `sacct`, and the log path printed at submission). A saved
checkpoint is an FSDP sharded model-and-optimizer checkpoint; retain the exact
Git commit, `model.lock`, `uv.lock`, dataset manifest, Slurm job ID, and logs
with every reported result.

To change batch size, context, optimizer, or checkpoint policy, edit the
training script on a branch and repeat the smoke test. Never compare runs that
changed both infrastructure and scientific settings without a matched control.

## Portability notes

- The launcher assumes homogeneous GPU counts and one `srun` task per node.
- `MASTER_ADDR` is the first allocated hostname; `MASTER_PORT` is derived from
  the Slurm job ID.
- NCCL interface selection is left to the site. Add site-approved `NCCL_*`
  variables to `config/site_setup.sh` only when the cluster documentation
  requires them.
- The code uses Transformers FSDP2. See the
  [Transformers FSDP documentation](https://huggingface.co/docs/transformers/v5.20.0/fsdp).
- An out-of-memory failure is not permission to silently enable LoRA,
  quantization, freezing, or CPU offload. Reduce context/microbatch, add GPUs,
  or document a separately validated full-parameter strategy.

## Repository map

```text
config/cluster.env.example   site-specific Slurm and path settings
scripts/submit_smoke.sh      topology + one-step acceptance submission
scripts/submit_train.sh      real training submission
scripts/slurm_job.sh         common in-allocation orchestration
scripts/distributed_entry.sh one torchrun per node
train_full_sft.py            full-parameter SFT implementation
nccl_probe.py                exact distributed-boundary proof
model.lock                   immutable model and validated/reference versions
```

Code in this repository is MIT licensed. Model and dataset licenses remain
independent.
