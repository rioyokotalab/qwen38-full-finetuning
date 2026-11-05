#!/usr/bin/env python3
"""Prove the complete requested multi-node NCCL boundary."""

import json
import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    expected_world = int(os.environ["EXPECTED_WORLD_SIZE"])
    expected_nodes = int(os.environ["EXPECTED_NODES"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    expected_sum = world * (world + 1) / 2
    if world != expected_world or value.item() != expected_sum:
        raise RuntimeError(
            f"collective-mismatch world={world}/{expected_world} "
            f"value={value.item()}/{expected_sum}"
        )
    identity = {
        "rank": rank,
        "host": socket.gethostname(),
        "local_rank": local_rank,
        "gpu": torch.cuda.get_device_name(local_rank),
        "memory": torch.cuda.get_device_properties(local_rank).total_memory,
    }
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(identity, gathered, dst=0)
    if rank == 0:
        hosts = {item["host"] for item in gathered}
        gpus = {(item["host"], item["local_rank"]) for item in gathered}
        if len(hosts) != expected_nodes or len(gpus) != expected_world:
            raise RuntimeError(
                f"topology-mismatch hosts={len(hosts)}/{expected_nodes} "
                f"gpus={len(gpus)}/{expected_world}"
            )
        print("NCCL_ACCEPTED " + json.dumps(gathered, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
