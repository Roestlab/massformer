#!/bin/bash

set -e

# demo data and parameters
download_scripts/download_demo.sh

# processed spectrum ID data
download_scripts/download_casmi_2016_proc.sh
download_scripts/download_casmi_2022_proc.sh
download_scripts/download_nist20_outlier_proc.sh

# raw data
download_scripts/download_mona_raw.sh
download_scripts/download_casmi_raw.sh

# CFM predictions on all data
download_scripts/download_cfm_pred.sh
