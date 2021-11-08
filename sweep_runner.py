import wandb
import os
import argparse
from functools import partial

from runner import init_or_resume_wandb_run, load_config

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-a","--account_name", type=str, default="adamoyoung")
	parser.add_argument("-p","--project_name", type=str, default="massformer-sweep")
	parser.add_argument("-s","--sweep_key", type=str, required=True)
	# parser.add_argument("-n","--run_name",type=str,required=True)
	parser.add_argument("-c","--count", type=int, default=1)
	parser.add_argument("-i","--job_id", type=int, required=True)
	parser.add_argument("-j","--job_id_dp", type=str, default="job_id")
	parser.add_argument("-t","--template_fp", type=str, default="config/template.yml")
	parser.add_argument("-m","--wandb_meta_dp", type=str, default=os.getcwd())
	flags = parser.parse_args()
	
	if flags.count < 0:
		count = 1000 # big number
	else:
		count = flags.count

	_, _, _, data_d, model_d, run_d = load_config(flags.template_fp,None,None)

	agent_func = partial(
		init_or_resume_wandb_run,
		project_name=flags.project_name,
		run_name=None,
		data_d=data_d,
		model_d=model_d,
		run_d=run_d,
		wandb_meta_dp=flags.wandb_meta_dp,
		job_id=flags.job_id,
		job_id_dp=flags.job_id_dp,
		is_sweep=True
	)

	job_fp = os.path.join(flags.job_id_dp,f"{flags.job_id}.yml")
	if os.path.exists(job_fp):
		agent_func()
	else:
		wandb.agent(
			flags.sweep_key,
			count=1,
			project=flags.project_name,
			function=agent_func
		)

