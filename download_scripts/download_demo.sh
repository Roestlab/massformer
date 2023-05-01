#!/bin/bash

# data
wget -O data/proc_demo.tgz https://zenodo.org/record/7874421/files/proc_demo.tgz?download=1
tar -xzf data/proc_demo.tgz -C data
rm data/proc_demo.tgz

# checkpoint
wget -O checkpoints/demo.pkl.gz https://zenodo.org/record/7874421/files/demo.pkl.gz?download=1
gunzip checkpoints/demo.pkl.gz

