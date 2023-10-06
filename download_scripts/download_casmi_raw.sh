#!/bin/bash

set -e

# CASMI 2016 data
wget -O data/raw/casmi_2016.tgz https://zenodo.org/record/8399738/files/casmi_2016.tgz?download=1
rm -rf data/raw/casmi_2016
tar -xzf data/raw/casmi_2016.tgz -C data/raw/
rm data/raw/casmi_2016.tgz

# CASMI 2022 data
wget -O data/raw/casmi_2022.tgz https://zenodo.org/record/8399738/files/casmi_2022.tgz?download=1
rm -rf data/raw/casmi_2022
tar -xzf data/raw/casmi_2022.tgz -C data/raw/
rm data/raw/casmi_2022.tgz

# CID->SMILES map
wget -O data/raw/cid_smiles.tsv.gz https://zenodo.org/record/8399738/files/cid_smiles.tsv.gz?download=1
rm -f data/raw/cid_smiles.tsv
gunzip data/raw/cid_smiles.tsv.gz
