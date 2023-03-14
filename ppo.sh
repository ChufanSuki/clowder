# `--nproc_per_node=2` specifies how many subprocesses we spawn for training with data parallelism
# note it is possible to run this with a *single GPU*: each process will simply share the same GPU
poetry install --with atari
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4

# by default we use the `gloo` backend, but you can use the `nccl` backend for better multi-GPU performance
poetry install --with atari
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --backend nccl

# it is possible to spawn more processes than the amount of GPUs you have via `--device-ids`
# e.g., the command below spawns two processes using GPU 0 and two processes using GPU 1
poetry install --with atari
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --env-id BreakoutNoFrameskip-v4 --device-ids 0 0 1 1