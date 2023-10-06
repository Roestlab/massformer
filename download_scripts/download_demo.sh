#!/bin/bash

set -e

# data
wget -O data/proc_demo.tgz https://zenodo.org/record/8399738/files/proc_demo.tgz?download=1
rm -rf data/proc_demo
tar -xzf data/proc_demo.tgz -C data
rm data/proc_demo.tgz

# checkpoint
wget -O checkpoints/demo.pkl.gz https://zenodo.org/record/8399738/files/demo.pkl.gz?download=1
rm -f checkpoints/demo.pkl
gunzip checkpoints/demo.pkl.gz

