# Project working agreement

For a new task run `python3 tools/task-start.py --context`. Resume a named task
with `python3 tools/task-select.py next-task --task QFT-NNN --context`.
`TODO.md` contains only open work; complete records move to
`docs/history/task-records/` and `docs/history/tasks.tsv`.

The repository must remain cluster-portable: never commit credentials, model
weights, datasets, logs, checkpoints, private host names, account IDs, or
absolute site paths. Site differences belong in ignored
`config/cluster.env` or the documented `config/site_setup.sh` hook.

This project means full-parameter BF16 fine-tuning. Do not silently introduce
LoRA/PEFT, quantization, frozen modules, CPU offload, or an unpinned model
revision. Treat scheduler acceptance as separate from NCCL acceptance and a
finite backward/optimizer step. Claims about speed, memory, or scientific
quality require matched empirical evidence.

Use `uv sync --locked` for the portable default. Changes to PyTorch, CUDA,
Transformers, FSDP policy, launcher topology, or model revision require a new
multi-node smoke run before publication. Keep `README.md` executable as a
handoff: every command must name its expected location and output.
