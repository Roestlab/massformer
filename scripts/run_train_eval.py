import argparse
import os
import numpy as np

from massformer.misc_utils import booltype, np_temp_seed
from massformer.runner import init_wandb_run, load_config, train_and_eval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="path to template config file")
    parser.add_argument(
        "-w",
        "--wandb_mode",
        type=str,
        default="off",
        choices=[
            "online",
            "offline",
            "disabled",
            "off"],
        help="wandb mode")
    parser.add_argument(
        "-d",
        "--device_id",
        type=int,
        required=False,
        help="device id (-1 for cpu)")
    parser.add_argument(
        "-c",
        "--custom_fp",
        type=str,
        required=False,
        help="path to custom config file")
    parser.add_argument(
        "-m",
        "--wandb_meta_dp",
        type=str,
        default=os.getcwd(),
        help="path to directory in which the wandb directory will exist")
    parser.add_argument(
        "-n", 
        "--num_seeds", 
        type=int, 
        default=0,
        help="number of random seeds to use")
    parser.add_argument(
        "-s", 
        "--reseed_splits", 
        type=booltype, 
        default=False,
        help="whether to reseed the splits (use a different split for each random seed)")
    parser.add_argument(
        "-g", 
        "--seed_idx", 
        type=int, 
        default=0,
        help="index of random seed to use")
    parser.add_argument(
        "-k", 
        "--checkpoint_name", 
        type=str, 
        required=False,
        help="name of checkpoint to load (from checkpoint_dp)")
    parser.add_argument(
        "-i", 
        "--job_id", 
        type=int, 
        required=False, 
        help="job_id for preemption")
    parser.add_argument(
        "-j", 
        "--job_id_dp", 
        type=str, 
        default="job_id", 
        help="directory where job_id files are stored")
    parser.add_argument(
        "-l", 
        "--wandb_symlink_dp", 
        type=str, 
        required=False,
        help="directory to store symlinks to wandb run directories")
    flags = parser.parse_args()

    use_wandb = flags.wandb_mode != "off"

    entity_name, project_name, run_name, data_d, model_d, run_d = load_config(
        flags.template_fp,
        flags.custom_fp,
        flags.device_id,
        flags.checkpoint_name
    )

    if use_wandb:

        if flags.num_seeds > 0:
            assert flags.num_seeds > 1, flags.num_seeds
            assert flags.seed_idx >= 0, flags.seed_idx
            # get the seeds
            if run_d["train_seed"] is None:
                meta_seed = 420420420
            else:
                meta_seed = run_d["train_seed"]
            with np_temp_seed(meta_seed):
                seed_range = np.arange(0, int(1e6))
                model_seeds = np.random.choice(
                    seed_range, replace=False, size=(
                        flags.num_seeds,))
                train_seeds = np.random.choice(
                    seed_range, replace=False, size=(
                        flags.num_seeds,))
                if flags.reseed_splits:
                    split_seeds = np.random.choice(
                        seed_range, replace=False, size=(
                            flags.num_seeds,))
                else:
                    split_seeds = np.array(
                        [run_d["split_seed"] for i in range(flags.num_seeds)])
                print("> model seeds:", model_seeds)
                print("> train seeds:", train_seeds)
                print("> split seeds:", split_seeds)
            group_name = f"{run_name}_rand"
            # only run the one with index seed_idx
            assert flags.seed_idx <= flags.num_seeds
            i = flags.seed_idx
            model_d["model_seed"] = model_seeds[i]
            run_d["train_seed"] = train_seeds[i]
            run_d["split_seed"] = split_seeds[i]
            run_d["cuda_deterministic"] = False
            run_name = f"{run_name}_{i}"
            if flags.job_id is not None:
                job_id_i = f"{flags.job_id}_{i}"
            else:
                job_id_i = None
            init_wandb_run(
                entity_name=entity_name,
                project_name=project_name,
                run_name=run_name,
                data_d=data_d,
                model_d=model_d,
                run_d=run_d,
                wandb_meta_dp=flags.wandb_meta_dp,
                group_name=group_name,
                wandb_mode=flags.wandb_mode,
                job_id=job_id_i,
                job_id_dp=flags.job_id_dp,
                wandb_symlink_dp=flags.wandb_symlink_dp
            )
        else:
            # just run one
            init_wandb_run(
                entity_name=entity_name,
                project_name=project_name,
                run_name=run_name,
                data_d=data_d,
                model_d=model_d,
                run_d=run_d,
                wandb_meta_dp=flags.wandb_meta_dp,
                wandb_mode=flags.wandb_mode,
                job_id=flags.job_id,
                job_id_dp=flags.job_id_dp,
                wandb_symlink_dp=flags.wandb_symlink_dp
            )

    else:

        train_and_eval(data_d, model_d, run_d, use_wandb)