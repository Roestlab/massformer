#!/bin/bash

wget -O data/cfm.tgz https://zenodo.org/record/7874421/files/cfm.tgz?download=1
mkdir data/tmp
tar -xzf data/cfm.tgz -C data/tmp
rm data/cfm.tgz
mv data/tmp/cfm/* data/cfm
rm -rf data/tmp
