#!/bin/bash

wget -O data/proc/casmi.tgz https://zenodo.org/record/7874421/files/casmi.tgz?download=1
tar -xzf data/proc/casmi.tgz -C data/proc/
rm data/proc/casmi.tgz
