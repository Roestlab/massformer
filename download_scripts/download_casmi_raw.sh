#!/bin/bash

# CASMI data
wget -O data/raw/casmi_2016.tgz https://zenodo.org/record/7874421/files/casmi_2016.tgz?download=1
tar -xzf data/raw/casmi_2016.tgz -C data/raw/

# CID->SMILES map
wget -O data/raw/cid_smiles.tsv.gz https://zenodo.org/record/7874421/files/cid_smiles.tsv.gz?download=1
gunzip data/raw/cid_smiles.tsv.gz
