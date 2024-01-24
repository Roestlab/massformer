import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import pandas as pd
import scipy
import scipy.stats
import json
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

import massformer.data_utils as data_utils
from massformer.plot_utils import viz_attention


# silly hack for sklearn
if sklearn.__version__.startswith("0."):
	NONE_PENALTY = "none"
else:
	NONE_PENALTY = None

ATOM_TO_NAME = {
	"N": "Nitrogen", 
	"Cl": "Chlorine", 
	"S": "Sulfur",
	"F": "Fluorine", 
	"P": "Phosphorus"
}

REDUCED_ELEMENT_LIST = [elem for elem in data_utils.ELEMENT_LIST if elem not in ["H","C"]]

def select_mol(ds,run_d,spec_ids,sims,rseed,min_num_peaks=0,sim_thresh=None,ce=None,prec_type=None,heteroatom="none",num_samples=4,return_group_ids=False):

	assert spec_ids is not None
	sim_df = pd.DataFrame({"spec_id":spec_ids,"sim":sims})
	df = ds.spec_df.merge(sim_df,on=["spec_id"],how="inner")
	df = df.merge(ds.mol_df,on=["mol_id"],how="inner")
	# filter for lots of peaks
	df = df[df["peaks"].apply(len)>min_num_peaks]
	if sims is not None and sim_thresh is not None:
		# filter for high similarity
		if isinstance(sim_thresh,tuple):
			assert len(sim_thresh) == 2
			df = df[(df["sim"]>=sim_thresh[0])&(df["sim"]<sim_thresh[1])]
		else:
			assert isinstance(sim_thresh,float)
			df = df[df["sim"]>=sim_thresh]
	if ce is not None:
		# filter for a particular collision energy
		df = df[df[ds.ce_key]==ce]
	if prec_type is not None:
		# filter for a particular precursor type
		df = df[df["prec_type"]==prec_type]
	if heteroatom != "none":
		# filter for molecules with heteroatoms
		if heteroatom == "all":
			heteroatom = "|".join(REDUCED_ELEMENT_LIST)
		df = df[df["formula"].str.contains(heteroatom)]
	if return_group_ids:
		group_df = df[["mol_id","group_id"]].drop_duplicates(subset=["mol_id"]).drop_duplicates(subset=["group_id"])
		assert group_df.shape[0] >= num_samples
		if num_samples > -1:
			# sample
			targ_group_id = group_df["group_id"].sample(n=num_samples,replace=False,random_state=rseed).values
		else:
			# get everything
			targ_group_id = group_df["group_id"].values
		return targ_group_id
	else:
		assert df["mol_id"].nunique() >= num_samples
		if num_samples > -1:
			# sample, without molecule duplicates
			group_df = df[["spec_id","mol_id"]].drop_duplicates(subset=["mol_id"])
			targ_spec_id = group_df["spec_id"].sample(n=num_samples,replace=False,random_state=rseed).values
		else:
			# get everything
			targ_spec_id = df["spec_id"].values
		return targ_spec_id

def compute_gi(model,data,gi_loss,nz_idx,input_feats,grad_viz_type):
	dummy_optim = th.optim.SGD(model.parameters(),lr=0.)
	dummy_optim.zero_grad()
	output_d = model(data,return_input_feats=input_feats)
	y = output_d["pred"]
	x_ = output_d["input_feats"]
	assert x_.shape[0] == 1
	x_.retain_grad()
	loss, mask_y = gi_loss(y,nz_idx)
	loss.backward()
	x = x_[0].detach()
	x_grad = x_.grad[0].detach()
	if grad_viz_type == "gi":
		gi = x*x_grad
	else:
		gi = x_grad
	dummy_optim.zero_grad()
	return gi, mask_y

def gi_loss_all(pred,no_pw):
	# import pdb; pdb.set_trace()
	if no_pw:
		pred = pred/(pred+EPS)
	return th.mean(pred), F.normalize(pred,p=1,dim=-1)

def gi_loss_single(pred,nz_idx,total,no_pw):
	# import pdb; pdb.set_trace()
	if no_pw:
		pred = pred/(pred+EPS)
	mask = F.one_hot(nz_idx,num_classes=total).float()
	return th.sum(pred*mask), F.normalize(pred,p=1,dim=-1)*mask

def reduce_and_norm(gi):
	# sum over embedding dimensions, then normalize over atoms
	gi = th.sum(gi,dim=1)
	gi = F.normalize(gi,p=2,dim=0)
	return gi

def create_cmap(colors):
	return mpl.colors.LinearSegmentedColormap.from_list("",colors)

def compute_gis(data,model,input_feats,pred=None,spec_bins=None,tree_bins=None,tree_labels=None,tree_formulae=None):
	
	# sometimes it's useful to precompute the prediction and spec_bins
	targ = data["spec"][0]
	if pred is None:
		with th.no_grad():
			pred = model(data,return_input_feats=False)["pred"][0]
	if spec_bins is None:
		spec_bins = th.nonzero((pred>0.).float()*(targ>0.).float(),as_tuple=True)[0].numpy()
	total = targ.shape[0]
	_gi_loss_single = lambda pred,nz_idx: gi_loss_single(pred,nz_idx,total,False)
	gis, gi_labels, gi_formulae, gi_bins, targ_vals = [], [], [], [], []
	for bin_idx, bin in enumerate(spec_bins):
		if tree_bins is None or bin not in tree_bins:
			gi_label = -1
			gi_formula = None
		else:
			tree_idx = np.where(tree_bins==bin)[0][0]
			if tree_labels is not None:
				gi_label = tree_labels[tree_idx]
			else:
				gi_label = -1
			if tree_formulae is not None:
				gi_formula = tree_formulae[tree_idx]
			else:
				gi_formula = None
		tbin = th.as_tensor(bin)
		gi, _ = compute_gi(model,data,_gi_loss_single,tbin,input_feats,"gi")
		gis.append(gi)
		gi_labels.append(gi_label)
		gi_formulae.append(gi_formula)
		gi_bins.append(bin)
		targ_vals.append(targ[bin].item())
	gis = th.stack(gis,dim=0).numpy()
	gi_labels = np.array(gi_labels)
	gi_formulae = np.array(gi_formulae)
	gi_bins = np.array(gi_bins)
	targ_vals = np.array(targ_vals)
	return gis, gi_labels, gi_formulae, gi_bins, targ_vals

def get_tree_data(tree_fp,heteroatom):

	tree = json.load(open(tree_fp,"r"))
	cur_tree_data = {}
	cur_tree_score = float(tree["annotations"]["statistics"]["explainedIntensity"])
	cur_tree_bins = []
	cur_tree_formulae = []
	cur_tree_labels = []
	for fragment in tree["fragments"]:
		cur_bin = int(np.floor(fragment["mz"]))
		cur_formula = fragment["molecularFormula"]
		cur_label = int(heteroatom in cur_formula)
		cur_tree_bins.append(cur_bin)
		cur_tree_formulae.append(cur_formula)
		cur_tree_labels.append(cur_label)
	# convert to arrays
	cur_tree_bins = np.array(cur_tree_bins)
	cur_tree_labels = np.array(cur_tree_labels)
	cur_tree_data["score"] = cur_tree_score
	cur_tree_data["bins"] = cur_tree_bins
	cur_tree_data["formulae"] = cur_tree_formulae
	cur_tree_data["labels"] = cur_tree_labels
	return cur_tree_data

def plot_annotated_pca(gis,gi_labels,gi_formulae,gi_bins,targ_vals,smiles,heteroatom,cmap):

	# fit PCA on all of the data
	gis_pca = F.normalize(th.as_tensor(gis.reshape(gis.shape[0],-1)),p=2,dim=1).numpy()
	pca = PCA(n_components=2)
	gis_pca = pca.fit_transform(gis_pca)
	# fit linear model on overlapping data
	assert not np.all(gi_labels==-1)
	both_mask = gi_labels!=-1
	both_gis_pca = gis_pca[both_mask]
	both_gi_labels = gi_labels[both_mask]
	both_gi_formulae = gi_formulae[both_mask]
	both_gi_bins = gi_bins[both_mask]
	both_targ_vals = targ_vals[both_mask]
	logreg = LogisticRegression(penalty=NONE_PENALTY,random_state=0)
	logreg.fit(both_gis_pca,both_gi_labels)
	accuracy = logreg.score(both_gis_pca,both_gi_labels)
	# assert accuracy==1., accuracy
	# plot
	fig = plt.figure(figsize=(14,10))
	# create all the axes
	pca_gs = gridspec.GridSpec(1,1)
	pca_gs.update(left=0.07,right=0.70,top=0.80,bottom=0.08) # 0.07
	pca_ax = fig.add_subplot(pca_gs[0,0])
	im_axs = []
	im_gs_1 = gridspec.GridSpec(1,3)
	im_gs_1.update(left=0.02,right=0.70,top=0.98,bottom=0.80)
	for i in range(3):
		im_ax = fig.add_subplot(im_gs_1[0,i])
		im_axs.append(im_ax)
	im_gs_2 = gridspec.GridSpec(5,1)
	im_gs_2.update(left=0.72,right=0.93,top=0.97,bottom=0.02)
	for i in range(5):
		im_ax = fig.add_subplot(im_gs_2[i,0])
		im_axs.append(im_ax)
	cbar_gs = gridspec.GridSpec(3,1)
	cbar_gs.update(left=0.93,right=0.95,top=0.96,bottom=0.02)
	cbar_ax = fig.add_subplot(cbar_gs[1,0]) # middle one
	# plot pca
	c_d = {0: "purple", 1: "green", -1: "black"}
	h_d = {"N": "Nitrogen", "Cl": "Chlorine", "P": "Phosphorus", "S": "Sulfur"}
	l_d = {0: f"No {h_d[heteroatom]}", 1: h_d[heteroatom], -1: "Unknown"}
	for gi_label in c_d.keys():
		gi_label_mask = gi_labels==gi_label
		gis_pca_mask = gis_pca[gi_label_mask]
		color = c_d[gi_label]
		label = l_d[gi_label]
		pca_ax.scatter(gis_pca_mask[:,0],gis_pca_mask[:,1],c=color,label=label,s=60)
	pca_legend = pca_ax.legend(fontsize=16)
	colors = list(c_d.values())
	for idx, text in enumerate(pca_legend.get_texts()):
		text.set_color(colors[idx])
	pca_ax.set_xlabel(f"PC1 ({int(np.around(100*pca.explained_variance_ratio_[0]))}% Variance Explained)",fontsize=18,labelpad=10)
	pca_ax.set_ylabel(f"PC2 ({int(np.around(100*pca.explained_variance_ratio_[1]))}% Variance Explained)",fontsize=18,labelpad=0)
	pca_ax.tick_params(axis='both', which='major', labelsize=14)
	# plot decision boundary
	b = logreg.intercept_[0]
	w1, w2 = logreg.coef_.T
	c = -b/w2
	m = -w1/w2
	xlim = pca_ax.get_xlim()
	ylim = pca_ax.get_ylim()
	xd = np.array([xlim[0], xlim[1]])
	yd = m*xd + c
	pca_ax.plot(xd,yd,'k',lw=1,ls='--')
	# restore limits
	pca_ax.set_xlim(xlim)
	pca_ax.set_ylim(ylim)
	# fill background color
	pca_ax.fill_between(xlim,(ylim[0],ylim[0]),(ylim[1],ylim[1]),color='gray',alpha=0.2)
	# annotate some points with formula
	# choose the 4 highest peaks positive and negative peaks
	pos_targ_vals = both_targ_vals*(both_gi_labels==1).astype(float)
	pos_annotate_idx = np.argsort(pos_targ_vals)[-4:]
	pos_annotate_idx = pos_annotate_idx[np.argsort(both_gi_bins[pos_annotate_idx])]
	neg_targ_vals = both_targ_vals*(both_gi_labels==0).astype(float)
	neg_annotate_idx = np.argsort(neg_targ_vals)[-4:]
	neg_annotate_idx = neg_annotate_idx[np.argsort(both_gi_bins[neg_annotate_idx])]
	all_annotate_idx = np.concatenate([pos_annotate_idx,neg_annotate_idx],axis=0)
	for i in range(all_annotate_idx.shape[0]):
		gi_idx = all_annotate_idx[i]
		gi_label = both_gi_labels[gi_idx]
		pos_x, pos_y = both_gis_pca[gi_idx]
		mass = both_gi_bins[gi_idx]-1
		annotation = f"{mass} Da"
		color = c_d[gi_label]
		if mass == 73:
			ha = "right"
			pos_x -= 0.02
			pos_y += 0.02
		elif mass == 97:
			ha = "left"
			pos_x += 0.02
			pos_y += 0.02
		elif mass == 115:
			ha = "right"
			pos_x -= 0.02
			pos_y -= 0.04
		elif mass == 259:
			ha = "left"
			pos_x += 0.02
			pos_y += 0.02
		elif mass == 140:
			ha = "left"
			pos_x += 0.02
			pos_y -= 0.04
		elif mass == 154:
			ha = "left"
			pos_x += 0.02
			pos_y += 0.02
		elif mass == 156:
			ha = "left"
			pos_x += 0.02
			pos_y -= 0.04
		elif mass == 182:
			ha = "left"
			pos_x += 0.02
			pos_y += 0.02
		else:
			# default if unknown
			ha = "center"
			color = "black"
		pca_ax.annotate(annotation,(pos_x,pos_y),ha=ha,color=color,fontsize=16)
	# plot images
	both_gis = gis[both_mask]
	both_gis_red = F.normalize(th.sum(th.as_tensor(both_gis.reshape(both_gis.shape[0],both_gis.shape[1],-1)),dim=2),p=2,dim=1).numpy()
	gi_svgs = []
	for idx, gi_idx in enumerate(all_annotate_idx):
		im_gi = both_gis_red[gi_idx]
		im_gi = (im_gi+1.)/2.
		gi_im = viz_attention(smiles,im_gi,cmap,1.0)
		gi_svg = viz_attention(smiles,im_gi,cmap,1.0,svg=True)
		gi_svgs.append(gi_svg)
		gi_im_arr = np.array(gi_im)
		im_ax = im_axs[idx]
		im_ax.imshow(gi_im_arr)
		im_ax.axis("off")
		position = (0.5,1.0) #(0.1,-0.2)
		annotation = f"{both_gi_bins[gi_idx]-1} Da - {both_gi_formulae[gi_idx]}"
		color = c_d[both_gi_labels[gi_idx]]
		im_ax.annotate(annotation,position,xycoords="axes fraction",ha="center",color=color,zorder=10,fontsize=16)
	iax = inset_axes(cbar_ax, width="50%", height="100%", loc="lower right", borderpad=0)
	cbar_ax.set_visible(False)
	norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)
	cm = mpl.colormaps[cmap]
	cb = mpl.colorbar.ColorbarBase(iax,cmap=cm,norm=norm,alpha=1.0,orientation="vertical")
	cb.ax.tick_params(labelsize=12)
	# print(cb.ax.get_yticks())
	cb.set_ticks([-1,-0.5,0.,0.5,1.0])
	cb.ax.set_yticklabels(["-1.0","-0.5"," 0.0"," 0.5"," 1.0"])
	return fig, gi_svgs, both_gi_bins[all_annotate_idx]

def get_hist(vals,bins):
	return np.histogram(vals,bins=bins,weights=np.zeros_like(vals)+1./len(vals))[0]

def plot_het_histogram(heteroatom,sep,sep_bl,acc,acc_bl,include_legend=True):

	het_color = "green"
	het_cmap = create_cmap(["white",het_color])
	bl_color = "sienna"
	bl_cmap = create_cmap(["white",bl_color])
	sep_expand = np.zeros((100,),dtype=np.float32)
	sep_expand[:int(np.around(sep*100))] = 1.
	acc_bl_avg = np.mean(acc_bl.reshape(len(acc),-1),axis=1)
	sep_bl_avg = np.mean(acc_bl.reshape(len(acc),-1)==1.,axis=1)
	# compute p-values
	acc_pval = ttest_ind(acc,acc_bl,equal_var=False,alternative="greater")[1]
	sep_pval = ttest_ind(acc==1.,acc_bl==1.,equal_var=False,alternative="greater")[1]
	acc_avg_pval = ttest_ind(acc,acc_bl_avg,equal_var=False,alternative="greater")[1]
	sep_avg_pval = ttest_ind(acc==1.,sep_bl_avg,equal_var=False,alternative="greater")[1]
	print(f">> {heteroatom} (N={len(acc)})")
	print(f"> acc: {acc_pval}")
	print(f"> sep: {sep_pval}")
	print(f"> acc_avg: {acc_avg_pval}")
	print(f"> sep_avg: {sep_avg_pval}")
	pval_df = pd.DataFrame(
		[
			{
				"acc": acc_pval,
				"sep": sep_pval,
				"acc_avg": acc_avg_pval,
				"sep_avg": sep_avg_pval
			}
		]
	)
	# compute histograms
	acc_bins = np.arange(0.5,1.01,0.025)
	acc_xlocs = np.arange(0.5,1.01,0.05)
	acc_xlabels = [f"{int(val)}%" for val in np.around(100*acc_xlocs)]
	acc_hist = get_hist(acc,acc_bins)
	acc_bl_hist = get_hist(acc_bl,acc_bins)
	# create figure
	fig = plt.figure(figsize=(15,15))
	hist_gs = gridspec.GridSpec(2,1)
	hist_gs.update(left=0.12,right=0.95,top=0.90,bottom=0.10)
	sep_arrow = dict(arrowstyle="-",linestyle="-",color="black",linewidth=3.0)
	mean_arrow = dict(arrowstyle="-",linestyle="dotted",color="black",linewidth=4.0)
	hist_axs = [fig.add_subplot(hist_gs[i]) for i in range(2)]
	hist_axs[0].hist(
		acc,
		bins=acc_bins,
		color=het_color,
		alpha=1.0,
		weights=np.zeros_like(acc)+1./len(acc),
		log=False,
		zorder=3
	)
	hist_axs[0].set_ylim(0.0,0.5)
	hist_axs[0].set_xlim(0.5-0.02,1.0+0.02)
	hist_axs[0].set_xticks(acc_xlocs,labels=acc_xlabels,fontsize=24)
	hist_axs[0].set_yticks([0.1,0.2,0.3,0.4,0.5],labels=["0.1","0.2","0.3","0.4","0.5"],fontsize=24)
	hist_axs[0].annotate(
		f"{100*sep:.0f}% Linearly Separable",
		xy=(0.975,0.5*acc_hist[-1]),
		xycoords="data",
		xytext=(0.45,0.70),
		textcoords="axes fraction",
		fontsize=30,
		color=het_color,
		ma="center",
		arrowprops=sep_arrow
	)
	hist_axs[0].annotate(
		f"Mean = {int(np.around(100*np.mean(acc)))}%",
		xy=(np.mean(acc),0.0),
		xycoords="data",
		xytext=(np.mean(acc),0.15),
		textcoords="data",
		fontsize=30,
		color=het_color,
		ha="center",
		arrowprops=mean_arrow,
		zorder=2
	)
	hist_axs[0].grid(
		visible=True,
		color="black",
		which="major",
		axis="y",
		linewidth=0.3,
		alpha=0.3
	)
	hist_axs[1].hist(
		acc_bl,
		bins=acc_bins,
		color=bl_color,
		alpha=1.0,
		weights=np.zeros_like(acc_bl)+1./len(acc_bl),
		log=False,
		zorder=3
	)
	hist_axs[1].set_ylim(0.0,0.5)
	hist_axs[1].set_xlim(0.5-0.02,1.0+0.02)
	hist_axs[1].set_xticks(acc_xlocs,labels=acc_xlabels,fontsize=24)
	hist_axs[1].set_yticks([0.1,0.2,0.3,0.4,0.5],labels=["0.1","0.2","0.3","0.4","0.5"],fontsize=24)
	hist_axs[1].annotate(
		f"{100*sep_bl:.0f}% Linearly Separable",
		xy=(0.975,0.5*acc_bl_hist[-1]),
		xycoords="data",
		xytext=(0.45,0.70),
		textcoords="axes fraction",
		fontsize=30,
		color=bl_color,
		ma="center",
		arrowprops=sep_arrow
	)
	hist_axs[1].annotate(
		f"Mean = {int(np.around(100*np.mean(acc_bl)))}%",
		xy=(np.mean(acc_bl),0.0),
		xycoords="data",
		xytext=(np.mean(acc_bl),0.15),
		textcoords="data",
		fontsize=30,
		color=bl_color,
		ha="center",
		arrowprops=mean_arrow,
		zorder=2
	)
	hist_axs[1].grid(
		visible=True,
		color="black",
		which="major",
		axis="y",
		linewidth=0.3,
		alpha=0.3
	)
	# titles
	fig.supylabel("Proportion of Spectra",fontsize=30,x=0.03, y=0.5)
	fig.supxlabel("Linear Classifier Accuracy",fontsize=30,x=0.54,y=0.02)
	# legend
	het_patch = mpatches.Patch(color=het_color,label=f'{ATOM_TO_NAME[heteroatom]} Labelling')
	bl_patch = mpatches.Patch(color=bl_color,label='Random Labelling')
	if include_legend:
		fig.legend(handles=[het_patch,bl_patch],loc="upper center",ncol=2,framealpha=1.0,fontsize=30,frameon=False)
	mpl_d = {
		"fig": fig,
		"hist_axs": hist_axs, 
		"pval_df": pval_df
	}
	return mpl_d
