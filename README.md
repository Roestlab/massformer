# MassFormer

This is the original implementation of MassFormer, a graph transformer for small molecule MS/MS prediction. Check out the preprint on [arxiv](https://arxiv.org/abs/2111.04824).

## System Requirements

This software requires a Unix-like operating system (we tested on Linux with Ubuntu 18.04).

For fast training/evaluation, a GPU compatible with [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) and at least 12GB VRAM is recommended.

We tested on an [Intel Xeon Silver 4110](https://ark.intel.com/content/www/us/en/ark/products/123547/intel-xeon-silver-4110-processor-11m-cache-2-10-ghz.html) system with an [Nvidia Tesla T4](https://www.nvidia.com/en-us/data-center/tesla-t4/) (16GB VRAM) and 64GB system RAM.

## Setting up the Environment

We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to create an environment, then installing the packages using [pip](https://pypi.org/project/pip/). The massformer code is configured to work with CUDA 11.3. Other versions of CUDA will likely work, but the install will require some modification. We also provide instructions for setting up the code to run on CPU only.

The total install time should not take more than a few minutes.

### GPU Environment

Enter the root directory and run the following commands:

```
cd massformer
conda create -n MF-GPU -y
conda activate MF-GPU
conda install python=3.8 -y
```

Then, install the dependencies using pip:

```
pip install -r env/requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html -f https://data.dgl.ai/wheels/cu113/repo.html
```

Finally, install massformer itself using pip:

```
pip install -I -e .
```

### CPU Environment

Enter the root directory and run the following commands:

```
cd massformer
conda create -n MF-CPU -y
conda activate MF-CPU
conda install python=3.8 -y
```

Then, install the dependencies using pip:

```
pip install -r env/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
```

Finally, install massformer itself using pip:

```
pip install -I -e .
```

## Downloading Everything

To download everything all at once, run the following script:

```
bash download_scripts/download_public.sh
```

The total size of all downloaded data is around 6.6GB. If you run this step, you can safely skip every step that starts with "Downloading" ([1](#downloading-demo-data-and-model-checkpoint), [2](#downloading-the-raw-data), [3](#downloading-the-casmipcasmi-processed-datasets), [4](#downloading-the-cfm-predictions)).

Otherwise, you may simply download files incrementally throughout the setup process.

## Downloading Demo Data and Model Checkpoint

To download the data and model checkpoint, run the following command:

```
bash download_scripts/download_demo.sh
```

## Running the Demo

The parameters for a pretrained MassFormer model (trained on the MoNA dataset, using an InChIKey split) are located in [checkpoints/demo.pkl](checkpoints/demo.pkl). We provide a script that loads these parameters and makes predictions on a heldout subset of the MoNA dataset.

### GPU Demo

To run the GPU demo, use the following command:

```
python src/massformer/runner.py -c config/demo.yml -w off -d 0
```

It should take 1-2 minutes to run.

### CPU Demo

To run the CPU demo, use the following command:

```
python src/massformer/runner.py -c config/demo.yml -w off -d -1
```

It should take 5-10 minutes to run (predictions are significantly slower without a GPU).

### Expected Output

The script should print the following statistics, in the form of a dictionary:

```
{'Epoch': 0,
 'epoch': 0,
 'test_m_mol_sim_obj_mean': tensor(0.5594),
 'test_m_spec_sim_obj_mean': tensor(0.5582),
 'test_mol_loss_obj_mean': tensor(0.4780),
 'test_mol_sim_obj_mean': tensor(0.5220),
 'test_spec_loss_obj_mean': tensor(0.4468),
 'test_spec_sim_obj_mean': tensor(0.5532)}
```
These metrics are recorded on a heldout portion of the MoNA dataset. `spec_loss_obj_mean` is the quantity that is minimized during training (log-transformed cosine similarity). `mol_loss_obj_mean` is the same quantity, but averaged over molecules instead of individual spectra. See the [config](config/demo.yml), the [loss definitions](src/massformer/losses.py), and the [runner script](src/massformer/runner.py) for more detailed information about these and other losses.

*Note: none of the steps below are required to run the demo, but are helpful for reproducing results from the paper.*

## Downloading the Raw Data

To obtain MassBank of North America and CASMI 2016 data, run the corresponding scripts for [mona](download_scripts/download_mona_raw.sh) and [casmi](download_scripts/download_casmi_raw.sh). These scripts will add files to the [data/raw](data/raw) directory.

```
bash download_scripts/download_mona_raw.sh
bash download_script/download_casmi_raw.sh
```

The [NIST 2020 LC-MS/MS dataset](https://www.nist.gov/programs-projects/nist20-updates-nist-tandem-and-electron-ionization-spectral-libraries) is not available for download directly, but can be purchased from an authorized distributor and exported using the instructions below.

## Exporting the NIST Data

*Note: this step requires a Windows System or Virtual Machine*

The spectra and associated compounds can be exported to MSP/MOL format using the free [lib2nist software](https://chemdata.nist.gov/mass-spc/ms-search/Library_conversion_tool.html). The resulting export will contain a single MSP file with all of the mass spectra, and multiple MOL files which include the molecular structure information (linked to the spectra by ID). The screenshot below indicates appropriate lib2nist export settings.

![Alt text](img/nist_export.png)

After exporting the files, create a directory "nist_20" in [data/raw](data/raw) and save them there. If done correctly, inside "nist_20" there should be a single .MSP file with all the spectra, hr_nist_msms.MSP, and a directory of .MOL files, hr_nist_msms.MOL.

There is a minor bug in the lib2nist export software that sometimes results in errors when parsing the MOL files. To fix this bug, run the following script:

```
python mol_fix.py --overwrite=True
``` 

This will edit the files in [data/raw/nist_20](data/raw/nist_20) and fix the bug.

## Preprocessing the Spectrum Datasets

To parse both of the datasets, run [parse_and_export.py](preproc_scripts/parse_and_export.py) using the following arguments:

### NIST  

```
python preproc_scripts/parse_and_export.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --output_name nist_df
```

### MoNA

```
python preproc_scripts/parse_and_export.py --msp_file mb_na_msms.msp --output_name mb_na_df
```

These scripts will create pandas dataframes (in json format) and save them in [data/df](data/df). Then, run [prepare_data.py](preproc_scripts/prepare_data.py) to convert these json dataframes in pickle dataframes that are used for model training and inference: 

```
python preproc_scripts/prepare_data.py
```

This will aggregate the NIST and MoNA data, and produce two pickle dataframes in [data/proc](data/proc): spec_df.pkl, which contains the spectrum information, and mol_df.pkl, which contains the structure information.

Both of these scripts using multiprocessing to speed up the computations: if your machine does not have many cores, it may be quite slow.

## Downloading the CASMI/pCASMI Processed Datasets

To obatain the CASMI/pCASMI datasets that were using for spectrum identification in the paper, run the corresponding scripts for [CASMI](download_scripts/download_casmi_proc.sh) and [pCASMI](download_scripts/download_pcasmi_proc.sh). These scripts will add directories to the [data/proc](data/proc) directory.

```
bash download_scripts/download_casmi_proc.sh
bash download_scripts/download_pcasmi_proc.sh
```

## Downloading the CFM Predictions

To reproduce the CFM evaluation (see Training and Evaluating a Model), the CFM predictions for the NIST/MoNA compounds must first be downloaded using the [CFM download script](download_scripts/download_cfm_pred.sh):

```
bash download_scripts/download_cfm_pred.sh
```

## Setting Up Weights and Biases

Our implementation uses [Weights and Biases (W&B)](https://wandb.ai/site) for logging and visualization. For full functionality (logging metrics, saving models), you must set up a free W&B account.

## Training and Evaluating a Model

The model configs for the experiments are stored in the [config](config/) directory. They are organized as follows:

- [all_prec_type](config/all_prec_type): models trained on NIST data with multiple adducts (i.e. Extended Data Figure 2)
- [mh_prec_type](config/mh_prec_type): models trained on NIST data with [M+H]+ adducts (i.e. Figure 2, Extended Data Figure 2)
- [casmi_pcasmi](config/casmi_pcasmi): models used for CASMI and pCASMI evaluations (i.e. Table 1, Table 2)
- [ablations](config/ablations): models used for ablation experiments (i.e. Extended Data Table 1)

Configurations exist of MassFormer (MF) and the baseline methods (FP, WLN, and CFM). 

To train and evluate a model, simply choose a configuration and pass it to the runner.py script. For example, to run the [mona_scaffold_all_MF](config/all_prec_type/mona_scaffold_MF.yml) experiment (train MassFormer on NIST data, and evaluate on MoNA using a scaffold split), you can use the following command: 

```
python src/massformer/runner.py -c config/all_prec_type/mona_scaffold_all_MF.yml -w online
```

The `-w` argument controls the wandb logging (online, offline, or off). Note that training a model without a GPU will be very time-consuming.

The configurations in [casmi_pcasmi](config/casmi_pcasmi) perform CASMI/pCASMI spectrum ID evaluations. The other configurations only record similarity on a held-out test set (NIST or MoNA, depending on the configuration).

## Accessing Pretrained Model Checkpoints

All models in the paper were train on NIST 20 data, which requires a license. To access these models, email ayoung [AT] cs [DOT] toronto [DOT] edu and verify your NIST license. Upon verification, you will be given access to the download link for the model checkpoints. 

Download these files and save them in the [checkpoints](checkpoints/) directory.

*Note: there are not any CFM checkpoints, since we use pre-computed CFM predictions (see [previous section](#training-and-evaluating-a-model)) in our experiments.*

## Evaluating a Pretrained Model

Assuming you have downloaded the checkpoints, you can evaluate a pretrained model using the [runner.py](src/massformer/runner.py) script. However, you need to modify a few things in the config file first:

- In the "model" section, add the following line that specifies which checkpoint to load
```
checkpoint_name: "<model_checkpoint_filename>"
```
- In the "run" section, change the "num_epochs" field to 0 (this prevents any training iterations).

Continuing the example from the [Training and Evaluating a Model](#training-and-evaluating-a-model) section, if you want to evaluate a pretrained [mona_scaffold_all_MF](config/all_prec_type/mona_scaffold_MF.yml) model using the mona_scaffold_all_MF_0.pkl checkpoint, you can modify the config to look like the following:

```
run_name:
data:
  num_entries: -1
  primary_dset: ["nist"]
  secondary_dset: ["mb_na"]
  ce_key: "nce"
  inst_type: ["FT"]
  frag_mode: ["HCD"]
  pos_prec_type: ['[M+H]+', '[M+H-H2O]+', '[M+H-2H2O]+', '[M+2H]2+', '[M+H-NH3]+', "[M+Na]+"]
  fp_types:   ["morgan","rdkit","maccs"]
  preproc_ce: "normalize"
  spectrum_normalization: "l1"
  casmi_num_entries: -1
  pcasmi_num_entries: -1
  gf_algos_v: "algos2"
model:
  embed_types: ["gf_v2"]
  ff_layer_type: "neims"
  ff_h_dim: 1000
  ff_num_layers: 4
  ff_skip: True
  dropout: 0.4
  output_normalization: "l1"
  bidirectional_prediction: True
  gf_model_name: "graphormer_base"
  gf_pretrain_name: "pcqm4mv2_graphormer_base"
  fix_num_pt_layers: 0
  reinit_num_pt_layers: -1
  reinit_layernorm: True
  model_seed: 6666
  checkpoint_name: mona_scaffold_MF_0 ### <--- NEW
run:
  loss: "cos"
  sim: "cos"
  lr: 0.001
  batch_size: 100
  clip_grad_norm: 5.0
  scheduler: "polynomial"
  scheduler_peak_lr: 0.0002
  weight_decay: 0.001
  pt_weight_decay: 0.0
  dp: False
  dp_num_gpus: 0
  flag: True
  num_epochs: 0 ### <--- NEW
  num_workers: 10
  cuda_deterministic: False
  do_test: True
  do_matching: False
  do_matching_2: False
  do_casmi: False
  do_pcasmi: False
  save_state: True
  save_media: True
  casmi_save_sim: True
  casmi_batch_size: 512
  casmi_num_workers: 10
  casmi_pred_all: True
  pcasmi_pred_all: False
  log_pcasmi_um: True
  log_auxiliary: True
  train_seed: 5585
  split_seed: 420
  split_key: "scaffold"
  log_tqdm: False
  grad_acc_interval: 4 # for 12GB
  test_sets: ["train","val","mb_na"]
  save_test_sims: True
  test_frac: 0.
```

Then, you can run the following command:

```
python src/massformer/runner.py -c config/all_prec_type/mona_scaffold_all_MF.yml -w online
```
