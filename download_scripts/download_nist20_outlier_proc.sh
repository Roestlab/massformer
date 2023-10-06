#!/bin/bash

set -e

wget -O data/proc/nist20_outlier.tgz https://zenodo.org/record/8399738/files/proc_nist20_outlier.tgz?download=1
rm -rf data/proc/nist20_outlier
tar -xzf data/proc/nist20_outlier.tgz -C data/proc/
rm data/proc/nist20_outlier.tgz
