#!/bin/bash

set -e

wget -O data/raw/mb_na_msms.msp.gz https://zenodo.org/record/8399738/files/mb_na_msms.msp.gz?download=1
rm -f data/raw/mb_na_msms.msp
gunzip data/raw/mb_na_msms.msp.gz
