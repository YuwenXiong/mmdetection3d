#!/bin/bash -i

source /root/.bashrc
export CPATH=/root/miniconda3/include:$CPATH
export CUDA_PATH=/root/miniconda3/include:$CUDA_PATH
conda init

cd $(dirname $0)/../..

git clone https://github.com/scaleapi/pandaset-devkit.git
cd pandaset-devkit/python
pip install .

cd $(dirname $0)/../..
git pull

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/root/miniconda3/bin/python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
