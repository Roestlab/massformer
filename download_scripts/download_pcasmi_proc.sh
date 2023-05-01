#!/bin/bash

wget -O data/proc/pcasmi.tgz https://zenodo.org/record/7874421/files/pcasmi.tgz?download=1
tar -xzf data/proc/pcasmi.tgz -C data/proc/
rm data/proc/pcasmi.tgz
