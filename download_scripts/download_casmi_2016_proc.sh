#!/bin/bash

set -e

wget -O data/proc/casmi_2016.tgz https://zenodo.org/record/8399738/files/proc_casmi_2016.tgz?download=1
rm -rf data/proc/casmi_2016
tar -xzf data/proc/casmi_2016.tgz -C data/proc/
rm data/proc/casmi_2016.tgz
