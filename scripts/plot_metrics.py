import wandb
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, FixedLocator
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse
import tempfile
import os
import time
import pandas as pd
from scipy.stats import ttest_ind, kstest

from massformer.metric_table import MetricTable
from massformer.misc_utils import booltype
from massformer.ont_parser import code_to_name
from massformer.plot_utils import plot_sim_hist

CASMI_TYPE_TO_NAME = {
	"casmi": "CASMI 2016",
	"pcasmi": "NIST20 Outlier",
	"casmi22": "CASMI 2022"
}

CASMI_TYPE_TO_COLOR = {
	"casmi": "tab:brown",
	"pcasmi": "tab:purple",
	"casmi22": "tab:olive"
}

def multi_welch_t_test(metrics,target_idx,alpha=0.01,alternative="greater"):

	target_metric = metrics[target_idx]
	# sidak correction
	alpha_corrected = 1.-(1.-alpha)**(len(metrics)-1)
	p_vals, rejects = [], []
	norm_p_vals = []
	for idx,metric in enumerate(metrics):
		# check for normality
		_, norm_p = kstest(metric,"norm")
		norm_p_vals.append(norm_p)
		if idx != target_idx:
			# check for difference of means
			_, p = ttest_ind(target_metric,metric,equal_var=False,alternative=alternative)
			p_vals.append(p)
			if np.isnan(p):
				rejects.append(False)
			else:
				rejects.append(p<alpha_corrected)
		else:
			p_vals.append(np.nan)
			rejects.append(True)
	return p_vals, rejects, norm_p_vals

def to_label(string):
	return " ".join(s.capitalize() for s in string.split("_"))

def plot_classyfire_sims(
		desc,
		prec_types,
		all_vals,
		all_mean,
		all_std,
		all_count,
		class_vals,
		class_means,
		class_stds,
		class_counts,
		class_names,
		output_dp,
		subset="frequency",
		size=32,
		y_min=0.0,
		y_max=1.0,
		individual_points=False,
		include_legend=True):

	if subset == "frequency":
		code_order = [k for k,v in sorted(class_counts.items(), key=lambda item: -item[1])][:10]
	else:
		raise NotImplementedError
	sns_colors = sns.color_palette("muted",n_colors=len(code_order))
	# sns_colors = sns.color_palette("rocket",n_colors=len(code_order))
	font_size = size
	tick_size = int(0.8*size)
	x_pad = int(size)
	y_pad = int(size)
	fig, ax = plt.subplots(figsize=(15,10),dpi=200)
	x_labels, x_labels_2, y_vals, y_means, y_errs, colors = [], [], [], [], [], []
	legend_d = {}
	for code_idx,code in enumerate(code_order):
		x_labels.append(class_names[code])
		x_labels_2.append(f"{(100*class_counts[code]/all_count):.1f}%")
		y_vals.append(class_vals[code])
		y_means.append(class_means[code])
		y_errs.append(class_stds[code])
		colors.append(sns_colors[code_idx])
		legend_d[x_labels[-1]] = colors[-1]
	x_pos = list(range(1,len(x_labels)+1))
	if individual_points:
		y_errs = None
	ax.bar(x_pos, y_means, yerr=y_errs, align='center', alpha=0.8, ecolor='black', capsize=10, color=colors)
	if individual_points:
		for pos, val in zip(x_pos, y_vals):
			val_poses = np.repeat(pos,len(val)).astype(float)
			if len(val) > 1:
				val_poses += np.random.uniform(-0.1,0.1,size=len(val))
			ax.scatter(
				val_poses,
				val,
				color="grey",
				marker="o",
				zorder=4.,
				alpha=0.5
			)
	# silly hack to get colors in legend
	for k,v in legend_d.items():
		ax.bar(x_pos, y_means, width=0, color=v, label=k)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(x_labels_2,rotation=0.)
	ax.set_ylim(bottom=y_min,top=y_max)
	ax.set_yticks(np.arange(y_min,y_max+0.05,0.05))
	ax.set_ylabel('Cosine Similarity',fontsize=font_size,labelpad=y_pad)
	ax.set_xlabel('ClassyFire Class Frequency',fontsize=font_size,labelpad=x_pad)
	ax.tick_params(axis="x", which="major", labelsize=0.9*tick_size)
	ax.tick_params(axis="y", which="major", labelsize=tick_size)
	ax.yaxis.grid(color="grey")
	ax.set_axisbelow(True)
	ax.axhline(all_mean,xmin=0,xmax=1,color="black",linestyle="dashed",linewidth=3.)
	if include_legend:
		ax.legend(fontsize=0.5*font_size,loc="upper left",framealpha=0.8)
	fig.tight_layout()
	model = desc.lower()
	ind_pt_str = "indpt" if individual_points else "dist"
	output_fp = os.path.join(output_dp,f"classyfire_{model}_{prec_types}_{subset}_{ind_pt_str}.png")
	fig.savefig(output_fp,bbox_inches="tight",format="png",dpi=300)
	fig.savefig(output_fp.replace(".png",".pdf"),bbox_inches="tight",format="pdf")
	plt.close()
	return None

def plot_both_boxplot_ranks(
		val_ds,
		desc_d,
		output_dp,
		casmi_types,
		rank_metric,
		size=36,
		logarithmic=False,
		include_legend=True):

	# model_order = ["CFM","FP","WLN","MF"]
	# model_to_idx = {model: idx for idx, model in enumerate(model_order)}
	assert len(val_ds) == len(casmi_types) == 3
	sns_colors = sns.color_palette("muted",n_colors=4)
	sns_colors = [sns_colors[3],sns_colors[0],sns_colors[1],sns_colors[2]]
	font_size = size
	tick_size = size
	x_pad = int(0.9*size)
	y_pad = int(size)
	linewidth = 6
	alpha = 1.0
	fig, ax = plt.subplots(figsize=(20,10),dpi=200)
	if rank_metric == "rank":
		if logarithmic:
			y_label = "Log10 Rank"
		else:
			y_label = "Rank" 
	else:
		if logarithmic:
			y_label = "Log10 Normalized Rank"
		else:
			y_label = "Normalized Rank"
	all_xs = [
		np.array([0.50,1.00,1.50,2.00]),
		np.array([2.75,3.25,3.75,4.25]),
		np.array([5.00,5.50,6.00,6.50]),
	]
	all_hatches = [None, "/", "//"]
	handleses = []
	for i in range(3):
		casmi_type = casmi_types[i]
		val_d = val_ds[i]
		ys = np.vstack([v for k,v in val_d.items()])
		models = [desc_d[k] for k in val_d.keys()]
		assert ys.shape[0] == len(models) == 4
		if logarithmic:
			if rank_metric == "rank":
				ys, ys_mean = np.log10(ys), np.log10(np.mean(ys,axis=1))
			else:
				ys, ys_mean = np.log10(1.+ys), np.log10(1.+np.mean(ys,axis=1))
		else:
			ys_mean = np.mean(ys,axis=1)
		xs = all_xs[i]
		box_hatch = all_hatches[i]
		labels = None
		handles = []
		for j in range(4):
			model_idx = j
			box_d = ax.boxplot(
				x=ys[model_idx:model_idx+1,:].T,
				positions=[xs[model_idx]],
				vert=True,
				labels=None,
				patch_artist=True,
				showcaps=False,
				boxprops={
					"alpha": alpha,
					"zorder": 3,
					"hatch": box_hatch,
					"linewidth": 3.0,
					"facecolor": sns_colors[model_idx]
				},
				showfliers=True,
				flierprops={
					"marker": "o",
					"zorder": 3,
					"linewidth": 3.0
				},
				whis=(10.,90.),
				whiskerprops={
					"linewidth": 3.0,
					"alpha": alpha,
					"zorder": 3
				},
				showmeans=False,
				zorder=3,
				medianprops={
					"linewidth": 3.0,
					"color": "black"
				},
				widths=[0.30]
			)
			ax.scatter(xs[model_idx],ys_mean[model_idx],marker="X",color="black",s=16**2,zorder=4)
			handle = mpatches.Patch(
				facecolor=sns_colors[model_idx],
				label=models[model_idx],
				alpha=alpha)
			handles.append(handle)
		handleses.append(handles)
	ax.set_ylabel(y_label,fontsize=font_size,labelpad=y_pad)
	ax.set_xlabel(None)
	ax.set_xticks([np.mean(all_xs[i]) for i in range(3)])
	ax.set_xticklabels([CASMI_TYPE_TO_NAME[casmi_type] for casmi_type in casmi_types],fontsize=font_size)
	ax.set_xlim(np.min(all_xs[0])-0.5,np.max(all_xs[-1])+0.5)
	if logarithmic and rank_metric == "rank":
		ax.set_ylim((-0.2,4.2))
	elif logarithmic and rank_metric == "norm_rank":
		ax.set_ylim((-0.01,0.21))
	elif not logarithmic and rank_metric == "norm_rank":
		ax.set_ylim((-0.02,0.52))
	ax.tick_params(axis="x", which="major", labelsize=font_size, length=0, pad=x_pad)
	ax.tick_params(axis="y", which="both", labelsize=tick_size)
	ax.grid(visible=True,which="both",axis="y",zorder=0)
	if include_legend:
		ax.legend(handles=handleses[0],loc="upper center",fontsize=font_size,ncol=4)
	fig.tight_layout()
	fig.savefig(os.path.join(output_dp,f"both_box_{rank_metric}.png"),format="png",dpi=300)
	fig.savefig(os.path.join(output_dp,f"both_box_{rank_metric}.pdf"),format="pdf")
	plt.close()
	return None

def plot_bar_sims(
		val_d,
		desc_d,
		output_dp,
		prec_types,
		cfm_filter,
		split_size_d,
		drop_cfm,
		merged,
		groupby_mol,
		size=24,
		individual_points=False,
		include_legend=True):

	if drop_cfm:
		assert cfm_filter == "none", cfm_filter
	sns_colors = sns.color_palette("muted",n_colors=4)
	hatches = [None,None,None,None]
	splits = ["NIST-InChIKey","MoNA-InChIKey","NIST-Scaffold","MoNA-Scaffold"]
	model_to_color = {
		"CFM": sns_colors[3],
		"FP": sns_colors[0],
		"WLN": sns_colors[1],
		"MF": sns_colors[2]
	}
	if drop_cfm:
		split_to_offset = {
			"NIST-InChIKey": 0.0,
			"MoNA-InChIKey": 3.0+0.5,
			"NIST-Scaffold": 3.0+0.5+3.0+0.5,
			"MoNA-Scaffold": 3.0+0.5+3.0+0.5+3.0+0.5
		}
		model_to_idx = {
			"FP": 0,
			"WLN": 1,
			"MF": 2
		}
	else:
		split_to_offset = {
			"NIST-InChIKey": 0.0,
			"MoNA-InChIKey": 4.0+0.5,
			"NIST-Scaffold": 4.0+0.5+4.0+0.5,
			"MoNA-Scaffold": 4.0+0.5+4.0+0.5+4.0+0.5
		}
		model_to_idx = {
			"CFM": 0,
			"FP": 1,
			"WLN": 2,
			"MF": 3
		}
	font_size = size
	tick_size = int(0.8*size)
	x_pad = int(0.7*size)
	y_pad = int(size)
	alpha = 1.0
	poses = [list() for split in splits]
	labelses = [list() for split in splits]
	meanses = [list() for split in splits]
	errses = [list() for split in splits]
	colorses = [list() for split in splits]
	valses = [list() for split in splits]
	for idx, (k,v) in enumerate(val_d.items()):
		model, split = desc_d[k].split(":")
		mean = v[0]
		err = v[1]
		vals = v[2]
		split_idx = splits.index(split)
		if drop_cfm and model == "CFM":
			continue
		labelses[split_idx].append(model)
		meanses[split_idx].append(mean)
		errses[split_idx].append(err)
		poses[split_idx].append(model_to_idx[model]+split_to_offset[split])
		colorses[split_idx].append(model_to_color[model])
		valses[split_idx].append(vals)
	# compute p values
	psymbolses = []
	metric_df, p_val_df = [], []
	for split_idx, split in enumerate(splits):
		labels = labelses[split_idx]
		mf_idx = labels.index("MF")
		wln_idx = labels.index("WLN")
		metrics = valses[split_idx]
		alpha_exp = 0
		reject = True
		while reject:
			alpha_exp += 1
			p_vals, rejects, norm_p_vals = multi_welch_t_test(
				metrics,
				mf_idx,
				alternative="greater",
				alpha=10**(-alpha_exp)
			)
			reject = rejects[wln_idx]
		p_val_symbol = f"p < 1e-{alpha_exp-1}"
		print(split,p_val_symbol,norm_p_vals)
		psymbolses.append(p_val_symbol)
		metric_df_row = {"split": split}
		for label, metric in zip(labels,metrics):
			metric_df_row[label] = tuple(metric.tolist())
		metric_df.append(metric_df_row)
		p_val_df_row = {"split": split}
		for label, norm_p_val in zip(labels,norm_p_vals):
			p_val_df_row[label] = norm_p_val
		p_val_df.append(p_val_df_row)
	metric_df = pd.DataFrame(metric_df)
	metric_df = metric_df[["split"]+list(model_to_idx.keys())]
	p_val_df = pd.DataFrame(p_val_df)
	p_val_df = p_val_df[["split"]+list(model_to_idx.keys())]
	print(p_val_df)
	# plot
	fig = plt.figure(figsize=(20,10),dpi=200)
	gs = gridspec.GridSpec(1,1)
	gs.update(left=0.05,right=0.95,top=0.95,bottom=0.18)
	ax = fig.add_subplot(gs[0,0])
	# p-value global params
	y, h, col = 0.68, 0.005, "black"
	for split_idx, split in enumerate(splits):
		# bars
		if not individual_points:
			yerr = errses[split_idx]
			error_kw=dict(elinewidth=2.,capthick=2.,capsize=10.,ecolor="black",zorder=5.)
		else:
			yerr = None
			error_kw = {}
		# yerr = errses[split_idx]
		# error_kw=dict(elinewidth=2.,capthick=2.,capsize=10.,ecolor="black",zorder=5.)
		ax.bar(
			poses[split_idx],meanses[split_idx],yerr=yerr,
			color=colorses[split_idx],hatch=hatches[split_idx],label=split,
			align="center",alpha=alpha,zorder=3.,
			error_kw=error_kw
		)
		if individual_points:
			for pos, val in zip(poses[split_idx], valses[split_idx]):
				val_poses = np.repeat(pos,len(val))
				if len(val) > 1:
					val_poses += np.random.uniform(-0.1,0.1,size=len(val))
				ax.scatter(
					val_poses,
					val,
					color="grey",
					marker="o",
					zorder=4.,
					alpha=0.5
				)
		y = meanses[split_idx][-1]+errses[split_idx][-1]+0.008
		# print(split,y)
		# p-values
		x1, x2 = poses[split_idx][-2], poses[split_idx][-1]
		ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
		ax.text((x1+x2)*.5, y+1.5*h, psymbolses[split_idx], ha='center', va='bottom', color=col, fontsize=0.8*font_size)
	# axes ticks/labels
	xtick_locs = [np.mean(pos) for pos in poses]
	ax.set_xticks(xtick_locs)
	xticklabels = []
	for split in splits:
		train_size = split_size_d[split]
		xticklabel = f"{split}\n(N={train_size})"
		xticklabels.append(xticklabel)
	ax.set_xticklabels(xticklabels,fontsize=font_size,rotation=0.)
	if cfm_filter == "cfm":
		# everyone performs better with CFM
		ax.set_ylim(0.45,0.75)
	else:
		ax.set_ylim(0.4,0.7)
	ax.set_ylabel("Cosine Similarity",fontsize=font_size,labelpad=y_pad)
	ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
	ax.tick_params(axis="x", which="major", labelsize=font_size, length=0, pad=x_pad)
	ax.tick_params(axis="y", which="both", labelsize=tick_size)
	ax.grid(axis="y",which="major",zorder=0.)
	merged_str = "m" if merged else "um"
	groupby_str = "mol" if groupby_mol else "spec"
	ind_pt_str = "indpt" if individual_points else "dist"
	if drop_cfm:
		if include_legend:
			handles = []
			for k,v in model_to_color.items():
				if k != "CFM":
					handles.append(mpatches.Patch(facecolor=v,label=k,alpha=alpha))
			fig.legend(handles=handles,fontsize=font_size,loc=(0.38,0.01),ncol=3,framealpha=1.0)
		plot_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_dropcfm_{merged_str}_{groupby_str}_{ind_pt_str}.png")
		metric_df_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_dropcfm_{merged_str}_{groupby_str}_metrics.csv")
		p_val_df_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_dropcfm_{merged_str}_{groupby_str}_pvals.csv")
	else:
		if include_legend:
			handles = [mpatches.Patch(facecolor=v,label=k,alpha=alpha) for k,v in model_to_color.items()]
			fig.legend(handles=handles,fontsize=font_size,loc=(0.33,0.01),ncol=4,framealpha=1.0)
		plot_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_{cfm_filter}_{merged_str}_{groupby_str}_{ind_pt_str}.png")
		metric_df_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_dropcfm_{merged_str}_{groupby_str}_metrics.csv")
		p_val_df_fp = os.path.join(output_dp,f"bar_sims_{prec_types}_dropcfm_{merged_str}_{groupby_str}_pvals.csv")
	fig.savefig(plot_fp,bbox_inches="tight",format="png",dpi=300)
	fig.savefig(plot_fp.replace(".png",".pdf"),bbox_inches="tight",format="pdf")
	metric_df.to_csv(metric_df_fp,index=False)
	p_val_df.to_csv(p_val_df_fp,index=False)
	plt.close()

def plot_bar_sims_single(val_d,desc_d,output_dp,size=32):

	color = sns.color_palette("muted",n_colors=1)[0]
	font_size = size
	tick_size = 0.8*size
	x_pad = int(size)
	y_pad = int(size)
	pos = []
	labels = []
	vals = []
	errs = []
	for idx, (k,v) in enumerate(val_d.items()):
		labels.append(desc_d[k])
		vals.append(v[0])
		errs.append(v[1])
		pos.append(idx+1.0)
	fig = plt.figure(figsize=(20,10),dpi=200)
	ax = fig.add_subplot(1,1,1)
	ax.bar(
		pos,vals,yerr=errs,
		color=color,label="NIST-Scaffold",
		align="center",alpha=1.0,
		ecolor='black',capsize=10,zorder=3.
	)
	ax.set_xticks(pos)
	ax.set_xticklabels(labels,rotation=0.)
	ax.set_ylim(0.5,0.75)
	ax.spines.right.set_visible(False)
	ax.spines.top.set_visible(False)
	ax.set_ylabel("Cosine Similarity",fontsize=font_size,labelpad=y_pad)
	ax.set_xlabel("Model Configuration",fontsize=font_size,labelpad=x_pad)
	ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
	ax.tick_params(axis="x", which="major", labelsize=tick_size)
	ax.tick_params(axis="y", which="both", labelsize=tick_size)
	ax.grid(axis="y",which="both",zorder=0.)
	ax.legend(fontsize=font_size,loc="upper center",ncol=1,framealpha=1.0)
	output_fp = os.path.join(output_dp,"bar_sims_ablations.png")
	fig.savefig(output_fp,bbox_inches="tight",format="png",dpi=300)
	fig.savefig(output_fp.replace(".png",".pdf"),bbox_inches="tight",format="pdf")
	plt.close()

def plot_casmi_metric(val_d,casmi_metric,casmi_type,output_dp,size=42):

	sns_colors = sns.color_palette("muted",n_colors=4)
	font_size = size
	tick_size = int(0.8*size)
	x_pad = int(0.7*size)
	y_pad = int(0.7*size)
	y_min = 0.
	if "top" in casmi_metric: #in ["top01","top05","top10","top01%","top05%","top10%"]:
		y_max = 1.
	else:
		y_max = None
	fig, ax = plt.subplots(figsize=(10,10),dpi=200)
	x_labels, y_means, y_errs, colors, x_pos = [], [], [], [], []
	y_vals = []
	for group_idx, (group, val) in enumerate(val_d.items()):
		x_labels.append(desc_d[group])
		y_means.append(val[0])
		y_errs.append(val[1])
		colors.append(sns_colors[group_idx])
		x_pos.append(group_idx+1)
		y_vals.append(val[2])
	# import pdb; pdb.set_trace()
	if casmi_metric in ["rank","norm_rank"]:
		alternative = "less"
	else:
		alternative = "greater"
	target_idx = x_labels.index("MF")
	p_vals, rejects, norm_p_vals = multi_welch_t_test(y_vals,target_idx,alternative=alternative)
	print(casmi_metric,p_vals,rejects,norm_p_vals)
	if casmi_type == "casmi":
		hatch = None
	elif casmi_type == "pcasmi":
		hatch = "/"
	else:
		assert casmi_type == "casmi22", casmi_type
		hatch = "//"
	ax.bar(
		x_pos, y_means, yerr=y_errs, 
		align='center', alpha=1.0, 
		capsize=10, color=colors, label=x_labels,
		hatch=hatch, error_kw=dict(elinewidth=4.,capthick=4.,capsize=10.,ecolor="black")
	)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(x_labels,rotation=0.)
	ax.set_ylim(bottom=y_min,top=y_max)
	if casmi_metric in ["top01","top05","top10","top01%","top05%","top10%"]:
		ylabel = "Fraction"
	elif casmi_metric in ["rank"]:
		ylabel = "Rank"
	elif casmi_metric in ["norm_rank"]:
		ylabel = "Normalized Rank"
	else:
		ylabel = None
	ax.set_ylabel(ylabel,fontsize=font_size,labelpad=y_pad)
	# ax.set_xlabel('Model',fontsize=font_size,labelpad=x_pad)
	ax.set_xlabel(None)
	ax.tick_params(axis="x", which="major", labelsize=tick_size, length=0., pad=x_pad)
	ax.tick_params(axis="y", which="major", labelsize=tick_size)
	ax.yaxis.grid(color="grey")
	ax.set_axisbelow(True)
	fig.tight_layout()
	output_fp = os.path.join(output_dp,f"{casmi_type}_{casmi_metric.replace('%','p')}.png")
	fig.savefig(output_fp,bbox_inches="tight",format="png",dpi=300)
	fig.savefig(output_fp.replace(".png",".pdf"),bbox_inches="tight",format="pdf")
	plt.close()

def cfm_filter_val(val,un_mol_id,cfm_mol_id,cfm_filter_flag,mol_df,filter_type):

	val = val.numpy()
	un_mol_id = un_mol_id.numpy()
	if cfm_filter_flag != "none" and filter_type == "scaffold":
		# get scaffolds
		un_scaffold = mol_df.loc[un_mol_id]["scaffold"]
		cfm_scaffold = mol_df.loc[cfm_mol_id]["scaffold"]
	if cfm_filter_flag == "cfm":
		if filter_type == "scaffold":
			cfm_mask = np.isin(un_scaffold,cfm_scaffold)
		else:
			cfm_mask = np.isin(un_mol_id,cfm_mol_id)
	elif cfm_filter_flag == "nocfm":
		if filter_type == "scaffold":
			cfm_mask = ~np.isin(un_scaffold,cfm_scaffold)
		else:
			cfm_mask = ~np.isin(un_mol_id,cfm_mol_id)
	else:
		assert cfm_filter_flag == "none"
		cfm_mask = np.ones_like(un_mol_id,dtype=bool)
	val = th.as_tensor(val[cfm_mask])
	un_mol_id = th.as_tensor(un_mol_id[cfm_mask])
	return val, un_mol_id

def export_to_latex(df,columns):

	def formatter(x):
		mean, std = x
		if mean > 1.:
			mean_str = f"{np.round(mean,decimals=1):.01f}"
			std_str = f"{np.round(std,decimals=1):.01f}"
		elif mean < 0.1:
			mean_str = f"{np.round(mean,decimals=3):.03f}".lstrip("0")
			std_str = f"{np.round(std,decimals=3):.03f}".lstrip("0")
		else:
			mean_str = f"{np.round(mean,decimals=2):.02f}".lstrip("0")
			std_str = f"{np.round(std,decimals=2):.02f}".lstrip("0")
		if std == 0.:
			fmt_str = "$" + mean_str + "$"
		else:
			fmt_str = "$" + mean_str + " \pm " + std_str + "$"
		return fmt_str
	df = df[columns] # select and reorder
	formatters = [formatter for column in columns]
	latex_str = df.to_latex(escape=False,index=True,formatters=formatters)
	return latex_str

def plot_sims_histograms(group,casmi_type,casmi_numqs,casmi_sims,casmi_sims2,casmi_numcs,output_dp):

	weights = 1. / (float(casmi_numqs) * np.repeat(casmi_numcs, casmi_numcs).astype(float))
	casmi_name = CASMI_TYPE_TO_NAME[casmi_type]
	casmi_color = CASMI_TYPE_TO_COLOR[casmi_type]
	sims_title = None #"Query-Candidate Spectrum Similarity" #f"{casmi_name} Candidate-Query Spectrum Similarity ({desc})"
	sims_fp = os.path.join(output_dp,f"{casmi_type}_{desc}_sims_hist.png")
	plot_sim_hist(casmi_sims,sim_type="cosine",weights=weights,title=sims_title,fp=sims_fp,legend=True,color=casmi_color,figsize=(9,4))
	sims2_title = None #"Query-Candidate Structure Similarity" #f"{casmi_name} Candidate-Query Structure Similarity"
	sims2_fp = os.path.join(output_dp,f"{casmi_type}_sims2_hist.png")
	plot_sim_hist(casmi_sims2,sim_type="tanimoto",weights=weights,title=sims2_title,fp=sims2_fp,legend=True,color=casmi_color,figsize=(9,4))
	total_title = None #"Candidate Count" #f"{casmi_name} Candidate Count"
	total_fp = os.path.join(output_dp,f"{casmi_type}_total_hist.png")
	# create total plot
	size = 20
	big_font_size = int(size)
	small_font_size = int(0.8*size)
	tick_size = int(0.7*size)
	x_pad = y_pad = int(0.6*size)
	fig, ax = plt.subplots(figsize=(9,4),dpi=200)
	weights = np.ones_like(casmi_numcs,dtype=float)/float(casmi_numcs.shape[0])
	bins = np.arange(0, 10200, 200)
	ax.hist(casmi_numcs, bins=bins, weights=weights, color=casmi_color)
	ax.axvline(
		x=np.mean(casmi_numcs),
		color="black",
		linestyle="dashed",
		linewidth=2,
		label=f"Mean={int(np.round(np.mean(casmi_numcs)))}")
	ax.set_xlabel("# Candidates",
		fontsize=small_font_size,
		labelpad=x_pad)
	ax.set_ylabel("Fraction",
		fontsize=small_font_size,
		labelpad=y_pad)
	ax.tick_params(axis="x", which="both", labelsize=tick_size)
	ax.tick_params(axis="y", which="both", labelsize=tick_size)
	# fig.suptitle(total_title,fontsize=big_font_size)
	ax.legend(
		loc="upper right",
		ncol=1,
		framealpha=1.0,
		fontsize=small_font_size)
	fig.tight_layout()
	fig.savefig(total_fp,format="png",dpi=300)
	fig.savefig(total_fp.replace(".png",".pdf"),format="pdf")
	plt.close("all")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--entity_name", 
		type=str, 
		required=True,
		help="wandb entity name, for downloading run files")
	parser.add_argument(
		"--project_name", 
		type=str, 
		required=True,
		help="wandb project name, for downloading run files")
	parser.add_argument(
		"--sim_type", 
		type=str, 
		default="cos_std",
		help="the similarity metric to use (default is cosine similarity, cos_std)")
	parser.add_argument(
		"--merged", 
		type=booltype, 
		default=True,
		help="whether to merge spectra by collision energy")
	parser.add_argument(
		"--groupby_mol", 
		type=booltype, 
		default=True,
		help="whether to aggregate similarity scores by molecule when averaging")
	parser.add_argument(
		"--output_dp", 
		type=str, 
		default="figs",
		help="path to directory for saving figures")
	parser.add_argument(
		"--casmi_type", 
		type=str, 
		default="all", 
		choices=["casmi","pcasmi","casmi22","all"],
		help="spectrum identification evaluations to plot with the 'casmi_metrics' function ('casmi' is CASMI 2016, 'casmi22' is CASMI 2022, 'pcasmi' is NIST20 Outlier, 'all' is everything)")
	parser.add_argument(
		"--function", 
		type=str, 
		required=True, 
		choices=["bar_sims","classyfire_sims","casmi_metrics","ablations"],
		help="which function to run: 'bar_sims' produces bar plots of similarities, 'classyfire_sims' produces bar plots of similarities with classyfire labels, 'casmi_metrics' produces spectrum identification bar plots and tables, 'ablations' produces ablation bar plots and tables")
	parser.add_argument(
		"--prec_types", 
		type=str, 
		default="all", 
		choices=["all","mh"],
		help="precursor adduct filtering: 'all' means all adducts, 'mh' means only [M+H]+")
	parser.add_argument(
		"--cfm_filter", 
		type=str, 
		default="none", 
		choices=["none","cfm","nocfm","all"],
		help="filtering based on overlap with CFM training set: 'none' means no filtering, 'cfm' means only CFM molecules, 'nocfm' means only non-CFM molecules, 'all' means all three")
	parser.add_argument(
		"--drop_cfm", 
		type=booltype, 
		default=False,
		help="whether to drop CFM from bar plots when using the 'bar_sims' function")
	args = parser.parse_args()

	path = f"{args.entity_name}/{args.project_name}"
	api = wandb.Api(timeout=30)

	os.makedirs(args.output_dp,exist_ok=True)

	if args.function == "bar_sims":

		if args.prec_types == "all":

			desc_d = {
				"nist_inchikey_all_CFM_rand": "CFM:NIST-InChIKey",
				"nist_inchikey_all_FP_rand": "FP:NIST-InChIKey",
				"nist_inchikey_all_WLN_rand": "WLN:NIST-InChIKey",
				"nist_inchikey_all_MF_rand": "MF:NIST-InChIKey",
				"nist_scaffold_all_CFM_rand": "CFM:NIST-Scaffold",
				"nist_scaffold_all_FP_rand": "FP:NIST-Scaffold",
				"nist_scaffold_all_WLN_rand": "WLN:NIST-Scaffold",
				"nist_scaffold_all_MF_rand": "MF:NIST-Scaffold",
				"mona_inchikey_all_CFM_rand": "CFM:MoNA-InChIKey",
				"mona_inchikey_all_FP_rand": "FP:MoNA-InChIKey",
				"mona_inchikey_all_WLN_rand": "WLN:MoNA-InChIKey",
				"mona_inchikey_all_MF_rand": "MF:MoNA-InChIKey",
				"mona_scaffold_all_CFM_rand": "CFM:MoNA-Scaffold",
				"mona_scaffold_all_FP_rand": "FP:MoNA-Scaffold",
				"mona_scaffold_all_WLN_rand": "WLN:MoNA-Scaffold",
				"mona_scaffold_all_MF_rand": "MF:MoNA-Scaffold",
			}

			split_size_d = {
				"NIST-InChIKey": 15472,
				"NIST-Scaffold": 16440,
				"MoNA-InChIKey": 19130,
				"MoNA-Scaffold": 10389,
			}

		else: # mh

			desc_d = {
				"nist_inchikey_mh_CFM_rand": "CFM:NIST-InChIKey",
				"nist_inchikey_mh_FP_rand": "FP:NIST-InChIKey",
				"nist_inchikey_mh_WLN_rand": "WLN:NIST-InChIKey",
				"nist_inchikey_mh_MF_rand": "MF:NIST-InChIKey",
				"nist_scaffold_mh_CFM_rand": "CFM:NIST-Scaffold",
				"nist_scaffold_mh_FP_rand": "FP:NIST-Scaffold", 
				"nist_scaffold_mh_WLN_rand": "WLN:NIST-Scaffold",
				"nist_scaffold_mh_MF_rand": "MF:NIST-Scaffold",
				"mona_inchikey_mh_CFM_rand": "CFM:MoNA-InChIKey",
				"mona_inchikey_mh_FP_rand": "FP:MoNA-InChIKey",
				"mona_inchikey_mh_WLN_rand": "WLN:MoNA-InChIKey",
				"mona_inchikey_mh_MF_rand": "MF:MoNA-InChIKey",
				"mona_scaffold_mh_CFM_rand": "CFM:MoNA-Scaffold",
				"mona_scaffold_mh_FP_rand": "FP:MoNA-Scaffold",
				"mona_scaffold_mh_WLN_rand": "WLN:MoNA-Scaffold",
				"mona_scaffold_mh_MF_rand": "MF:MoNA-Scaffold", 
			}

			split_size_d = {
				"NIST-InChIKey": 13648,
				"NIST-Scaffold": 12405,
				"MoNA-InChIKey": 16812,
				"MoNA-Scaffold": 9035,
			}

	elif args.function == "classyfire_sims":
	
		if args.prec_types == "all":

			desc_d = {
				"nist_scaffold_all_CFM_rand": "CFM",
				"nist_scaffold_all_FP_rand": "FP",
				"nist_scaffold_all_WLN_rand": "WLN",
				"nist_scaffold_all_MF_rand": "MF",
			}

		else:

			desc_d = {
				"nist_scaffold_mh_CFM_rand": "CFM",
				"nist_scaffold_mh_FP_rand": "FP",
				"nist_scaffold_mh_WLN_rand": "WLN",
				"nist_scaffold_mh_MF_rand": "MF",
			}

	elif args.function == "casmi_metrics":

		desc_d = {
			"casmi_CFM_rand": "CFM",
			"casmi_FP_rand": "FP",
			"casmi_WLN_rand": "WLN",
			"casmi_MF_rand": "MF",
		}

	elif args.function == "ablations":

		desc_d = {
			"ablations2_small_rand": "Small",
			"ablations2_base_rand": "Large",
			"ablations2_pretrain_rand": "Large+PT",
			"ablations2_pretrain_reinit_rand": "Large+PT\n+TL",
			"ablations2_pretrain_ln_rand": "Large+PT\n+LN",
			"ablations2_pretrain_reinit_ln_rand": "Large+PT\n+TL+LN"
		}

	if args.function in ["bar_sims","classyfire_sims","ablations"]:

		if args.merged:
			table_sim = "m_sim"
		else:
			table_sim = "sim"

		metric_table_d = {}
		# create temporary directory
		with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dp:
			# cycle through replicates, download metric tables
			for group, desc in desc_d.items():
				if args.function == "bar_sims" and "MoNA" in desc:
					table_fn = f"mb_na_{table_sim}"
				else:
					table_fn = f"test_{table_sim}"
				print(f"> {group}")
				runs = api.runs(path=path,filters={"group":group})
				# assert all(run.state == "finished" for run in runs)
				metric_tables = []
				non_outlier_count = 0
				for run_idx,run in enumerate(runs):
					# if non_outlier_count > 1:
					# 	break
					if run.state not in ["finished","failed"]:
						print(">> not finished!!")
						print(run.state)
						continue
					print(f"> run {run_idx}")
					file = run.file(f"save_tables/{table_fn}.pt")
					if file.size == 0:
						print(">> empty!!")
					elif "outlier" in run.tags:
						print(">> outlier!!")
					else:
						file.download(root=tmp_dp,replace=True)
						metric_table = MetricTable.load(os.path.join(tmp_dp,f"save_tables/{table_fn}.pt"))
						metric_tables.append(metric_table)
						non_outlier_count += 1
				metric_table_d[group] = metric_tables

	if args.function in ["bar_sims","ablations"]:

		if args.function == "bar_sims":
			mol_df = pd.read_pickle("data/proc/mol_df.pkl")
		else:
			mol_df = None
		cfm_mol_id = th.as_tensor(pd.read_csv("data/cfm/all_mol_id.csv")["mol_id"].to_numpy())
		if args.drop_cfm:
			assert args.cfm_filter == "none", args.cfm_filter
		if args.cfm_filter == "all":
			cfm_filters = ["none","cfm","nocfm"]
		else:
			cfm_filters = [args.cfm_filter]
		table_d = {}
		for cfm_filter in cfm_filters:
			bar_d = {}
			for group, metric_tables in metric_table_d.items():
				vals = []
				un_mol_ids = []
				if args.function == "bar_sims" and "Scaffold" in desc_d[group]:
					filter_type = "scaffold"
				else:
					filter_type = "inchikey"
				for metric_table in metric_tables:
					val, un_mol_id = metric_table.get_val_mol_id(args.sim_type,groupby_mol=args.groupby_mol)
					val, un_mol_id = cfm_filter_val(val,un_mol_id,cfm_mol_id,cfm_filter,mol_df,filter_type)
					argsort_idx = th.argsort(un_mol_id)
					vals.append(val[argsort_idx])
					un_mol_ids.append(un_mol_id[argsort_idx])
				vals = th.stack(vals,dim=0)
				un_mol_ids = th.stack(un_mol_ids,dim=0)
				vals = th.mean(vals,dim=1)
				# print(cfm_filter,filter_type,vals.shape,un_mol_ids.shape)
				mean_val = th.mean(vals,dim=0).item()
				std_val = th.std(vals,dim=0).item()
				if np.isnan(std_val):
					std_val = 0.
				bar_d[group] = (mean_val,std_val,vals.numpy())
				table_d[(group,cfm_filter)] = (mean_val,std_val)
			if args.function == "bar_sims":
				output_dp = os.path.join(args.output_dp,"bar_sims")
				os.makedirs(output_dp,exist_ok=True)
				print(bar_d,desc_d)
				for ind_pt_flag in [False, True]:
					plot_bar_sims(
						bar_d,
						desc_d,
						output_dp,
						args.prec_types,
						cfm_filter,
						split_size_d,
						args.drop_cfm,
						args.merged,
						args.groupby_mol,
						individual_points=ind_pt_flag,
						include_legend=False
					)
			else:
				assert args.function == "ablations"
				print(bar_d)
				output_dp = os.path.join(args.output_dp,"ablations")
				plot_bar_sims_single(bar_d,desc_d,output_dp)

	if args.function == "classyfire_sims":

		# load classyfire meta
		classyfire_df = pd.read_csv("data/classyfire/all_classyfire_df.csv")
		classyfire_codes = sorted(list(set(classyfire_df["superclass_code"].dropna())))
		classyfire_names = [code_to_name[code] for code in classyfire_codes]
		class_names = {code:code_to_name[code] for code in classyfire_codes}

		val_d = {}
		for group, metric_tables in metric_table_d.items():
			all_vals = []
			all_count = None
			class_vals = {}
			class_counts = {}
			for metric_table in metric_tables:
				val, un_mol_id = metric_table.get_val_mol_id(args.sim_type,groupby_mol=args.groupby_mol)
				val_df = pd.DataFrame({"sim":val.numpy(),"mol_id":un_mol_id.numpy()})
				val_df = val_df.merge(classyfire_df,on=["mol_id"],how="inner")
				vals = val_df["sim"].to_numpy()
				all_vals.append(vals)
				means = []
				if all_count is None:
					all_count = val.shape[0]
				else:
					assert all_count == val.shape[0], (all_count,val.shape[0])
				# groupby class
				for cf_code in classyfire_codes:
					cf_df = val_df[val_df["superclass_code"]==cf_code]
					cf_vals = cf_df["sim"].to_numpy()
					if cf_code not in class_vals:
						class_vals[cf_code] = [cf_vals]
					else:
						class_vals[cf_code].append(cf_vals)
					if cf_code not in class_counts:
						class_counts[cf_code] = cf_df.shape[0]
					else:
						assert class_counts[cf_code] == cf_df.shape[0]
			all_vals = np.mean(np.stack(all_vals),axis=1)
			all_mean = np.mean(all_vals,axis=0)
			all_std = np.std(all_vals,axis=0)
			class_means = {}
			class_stds = {}
			for cf_code in classyfire_codes:
				class_vals[cf_code] = np.mean(np.stack(class_vals[cf_code]),axis=1)
				class_means[cf_code] = np.mean(class_vals[cf_code],axis=0)
				class_stds[cf_code] = np.std(class_vals[cf_code],axis=0)
			val_d[group] = (
				all_vals,
				all_mean,
				all_std,
				all_count,
				class_vals,
				class_means,
				class_stds,
				class_counts
			)
		output_dp = os.path.join(args.output_dp,"classyfire_sims")
		os.makedirs(output_dp,exist_ok=True)
		for group, vals in val_d.items():
			all_vals, all_mean, all_std, all_count = vals[:4]
			class_vals, class_means, class_stds, class_counts = vals[4:]
			# if desc_d[group] == "CFM":
			# 	if args.prec_types == "all":
			# 		y_min = 0.2
			# 	else:
			# 		y_min = 0.3
			# else:
			# 	y_min = 0.5
			y_min, y_max = 0.35, 0.75
			for ind_pt_flag in [False, True]:
				plot_classyfire_sims(
					desc_d[group],
					args.prec_types,
					all_vals,all_mean,all_std,all_count,
					class_vals,class_means,class_stds,class_counts,class_names,
					output_dp,
					y_min=y_min,
					y_max=y_max,
					individual_points=ind_pt_flag,
					include_legend=False
				)

	if args.function == "casmi_metrics":

		output_dp = os.path.join(args.output_dp,"casmi_metrics")
		os.makedirs(output_dp,exist_ok=True)
		if args.casmi_type == "all":
			casmi_types = ["casmi","casmi22","pcasmi"]
		else:
			casmi_types = [args.casmi_type]
		both_rank_ds, both_norm_rank_ds = [], []
		for casmi_type in casmi_types:
			print(f">>> {casmi_type}")
			casmi_fn = f"{casmi_type}_sims.pkl"
			casmi_res_d = {}
			# create temporary directory
			with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dp:
				# cycle through replicates, download metric tables
				for group, desc in desc_d.items():
					print(f"> {group}")
					runs = api.runs(path=path,filters={"group":group})
					assert all(run.state == "finished" for run in runs)
					casmi_reses = []
					non_outlier_count = 0
					for run_idx,run in enumerate(runs):
						# if non_outlier_count > 2:
						# 	break
						print(f"> run {run_idx}")
						file = run.file(casmi_fn)
						if file.size == 0:
							print(">> empty!!")
						elif "outlier" in run.tags:
							print(">> outlier!!")
						else:
							file.download(root=tmp_dp,replace=True)
							casmi_res = th.load(os.path.join(tmp_dp,casmi_fn))
							casmi_reses.append(casmi_res)
							non_outlier_count += 1
					casmi_res_d[group] = casmi_reses
			# plot histograms
			for group, casmi_reses in casmi_res_d.items():
				# average sims
				casmi_numqs, casmi_sims, casmi_sims2, casmi_numcs = [], [], [], []
				for casmi_res in casmi_reses:
					casmi_numqs.append(len(casmi_res["sims"]))
					casmi_sims.append(np.concatenate(casmi_res["sims"],axis=0).flatten())
					casmi_sims2.append(np.concatenate(casmi_res["sims2"],axis=0).flatten())
					casmi_numcs.append(np.array([sims.shape[0] for sims in casmi_res["sims"]]))
				casmi_numqs = casmi_numqs[0]
				casmi_sims = np.mean(np.stack(casmi_sims,axis=0),axis=0)
				casmi_sims2 = np.mean(np.stack(casmi_sims2,axis=0),axis=0)
				casmi_numcs = np.stack(casmi_numcs,axis=0)[0]
				desc = desc_d[group]
				plot_sims_histograms(desc,casmi_type,casmi_numqs,casmi_sims,casmi_sims2,casmi_numcs,output_dp)
			casmi_metrics = [
				"sim",
				"rank",
				"top01",
				"top05",
				"norm_rank",
				"top01%",
				"top05%",
				"t20p_rank",
				"t20p_top01",
				"t20p_top05",
				"b20p_rank",
				"b20p_top01",
				"b20p_top05"
			]
			casmi_metrics_df = {casmi_metric: list() for casmi_metric in casmi_metrics}
			casmi_metrics_df["model"] = [desc for group, desc in desc_d.items()]
			for casmi_metric in casmi_metrics:
				val_d = {}
				for group, casmi_reses in casmi_res_d.items():
					casmi_ranks = []
					for casmi_res in casmi_reses:
						casmi_ranks.append(casmi_res["rm_d"][f"{casmi_type}_{casmi_metric}"])
					casmi_ranks = np.stack(casmi_ranks,axis=0)
					mean_casmi_rank = np.mean(casmi_ranks,axis=1)
					val_d[group] = (np.mean(mean_casmi_rank),np.std(mean_casmi_rank),mean_casmi_rank)
					casmi_metrics_df[casmi_metric].append((np.mean(mean_casmi_rank),np.std(mean_casmi_rank)))
				# plot_casmi_metric(val_d,casmi_metric,casmi_type,output_dp)
				if casmi_metric in ["rank","norm_rank"]:
					val_d = {}
					for group, casmi_reses in casmi_res_d.items():
						casmi_ranks = []
						for casmi_res in casmi_reses:
							casmi_ranks.append(casmi_res["rm_d"][f"{casmi_type}_{casmi_metric}"])
						casmi_ranks = np.stack(casmi_ranks,axis=0)
						mean_casmi_rank = np.mean(casmi_ranks,axis=0)
						val_d[group] = mean_casmi_rank
					if casmi_metric == "rank":
						both_rank_ds.append(val_d)
					else:
						both_norm_rank_ds.append(val_d)
			casmi_metrics_df = pd.DataFrame(casmi_metrics_df)
			casmi_metrics_df = casmi_metrics_df.set_index("model",drop=True)
			# rearrange rows
			casmi_metrics_df = casmi_metrics_df.loc[["FP","WLN","MF","CFM"]]
			print(casmi_metrics_df)
			table_metric_groups = [
				["sim","rank","top01","top05","norm_rank","top01%","top05%"],
				["t20p_rank","t20p_top01","t20p_top05","b20p_rank","b20p_top01","b20p_top05"],
			]
			for table_idx, table_metric_group in enumerate(table_metric_groups):
				casmi_metrics_table = export_to_latex(casmi_metrics_df,table_metric_group)
				with open(os.path.join(output_dp,f"{casmi_type}_metrics_table_{table_idx}.txt"),"w") as file:
					file.write(casmi_metrics_table)

		if args.casmi_type == "all":
			plot_both_boxplot_ranks(
				both_rank_ds,
				desc_d,
				output_dp,
				casmi_types,
				"rank",
				logarithmic=True,
				include_legend=False)
			plot_both_boxplot_ranks(
				both_norm_rank_ds,
				desc_d,
				output_dp,
				casmi_types,
				"norm_rank",
				logarithmic=False,
				include_legend=False)

