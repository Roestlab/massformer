#!/bin/bash

set -e

wget -O data/cfm.tgz https://zenodo.org/record/8399738/files/cfm.tgz?download=1
rm -rf data/tmp
mkdir data/tmp
tar -xzf data/cfm.tgz -C data/tmp
rm data/cfm.tgz
rm -rf data/cfm/*
mv data/tmp/cfm/* data/cfm
rm -rf data/tmp
