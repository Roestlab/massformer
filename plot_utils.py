import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import io
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
from scipy.stats import gaussian_kde
import seaborn as sns


EPS = np.finfo(np.float32).eps

def trim_img_by_white(img, padding=0):
	'''
	This function takes a PIL image, img, and crops it to the minimum rectangle 
	based on its whiteness/transparency. 5 pixel padding used automatically.
	Adapted from:
	https://github.com/connorcoley/retrosim/blob/master/retrosim/utils/draw.py
	Copyright (c) 2017 Connor Coley
	See licenses/RETROSIM_LICENSE
	'''

	# Convert to array
	as_array = np.array(img)  # N x N x (r,g,b,a)
	assert as_array.ndim == 3 and as_array.shape[2] == 3, as_array.shape
	# Content defined as non-white and non-transparent pixel
	has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
	xs, ys = np.nonzero(has_content)
	# Crop down
	x_range = max([min(xs)-5,0]), min([max(xs)+5, as_array.shape[0]])
	y_range = max([min(ys)-5,0]), min([max(ys)+5, as_array.shape[1]])
	as_array_cropped = as_array[x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]
	img = Image.fromarray(as_array_cropped, mode='RGB')
	return ImageOps.expand(img, border=padding, fill=(255, 255, 255, 0))

def get_mol_im(smiles):

	width = 1000
	height = 1000
	mols = [Chem.MolFromSmiles(smiles)]
	d = rdMolDraw2D.MolDraw2DCairo(width,height)
	d.DrawMolecules(mols)
	d.FinishDrawing()
	png_buf = d.GetDrawingText()
	im = Image.open(io.BytesIO(png_buf))
	im = trim_img_by_white(im,padding=15)
	return im

def plot_spec(
	true_spec,
	gen_spec,
	mz_max,
	mz_res,
	prec_mz_bin=None,
	rescale_mz_axis=True,
	rescale_ints_axis=True,
	smiles=None,
	cfm_mbs=None,
	simple_mbs=None,
	plot_title=True,
	mol_image=None,
	loss_type=None,
	loss=None,
	sim_type=None,
	sim=None,
	height_ratios=None,
	size=24):
	"""
	Generates a plot comparing a true and predicted mass spec spectra.
	Adapted from: https://github.com/brain-research/deep-molecular-massspec
	Copyright (c) 2018 Google LLC
	See licenses/NEIMS_LICENSE
	"""

	bar_width = 0.8*mz_res

	if rescale_mz_axis:
		true_mz_max = np.flatnonzero(true_spec)[-1]*mz_res
		# gen_mz_max = np.flatnonzero(gen_spec)[-1]*mz_res
		# mz_max = max(true_mz_max, gen_mz_max)
	x_max = 1.05*min(mz_max,true_mz_max)
	x_array = np.arange(0.,x_max,step=mz_res)
	true_spec = true_spec.flatten()[:x_array.shape[0]]
	gen_spec = gen_spec.flatten()[:x_array.shape[0]]

	if np.max(true_spec) == 0.:
		import pdb; pdb.set_trace()

	if rescale_ints_axis:
		# ints_max = max(np.max(true_spec),np.max(gen_spec))
		ints_max = np.max(true_spec)
	else:
		ints_max = 1.
	y_max = 1.05*ints_max

	fig = plt.figure(figsize=(20,20), dpi=200)
	if plot_title:
		assert not (loss_type is None)
		assert not (sim_type is None)
		assert not (loss is None)
		assert not (sim is None)
		fig.suptitle(f"{loss_type}_loss = {loss:.4g}, {sim_type}_sim = {sim:.4g}",fontsize=20,y=0.925)
		if height_ratios is None:
			height_ratios = [1,3,3]
	else:
		if height_ratios is None:
			height_ratios = [1,2,2]
	ylabel_y = height_ratios[-1]/sum(height_ratios)

	# Adding extra subplot so both plots have common x-axis and y-axis labels
	y_pad = int(2.0*size)
	x_pad = int(size)
	tick_size = int(0.8*size)
	font_size = int(size)
	ax_main = fig.add_subplot(111, frameon=False)
	ax_main.tick_params(axis="both", which="major", labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax_main.set_xlabel('Mass/Charge (m/z)',fontsize=font_size,labelpad=x_pad)
	ax_main.set_ylabel('Relative Intensity',fontsize=font_size,y=ylabel_y,labelpad=y_pad)
	# ax_main.text(0.8,0.8,"{0:.4f}".format(loss_type))

	# set up gridspec
	gs = matplotlib.gridspec.GridSpec(3,1,height_ratios=height_ratios)

	assert not (smiles is None) or not (mol_image is None)
	if not (mol_image is None):
		mol_im = mol_image
	else:
		mol_im = get_mol_im(smiles)
	mol_im_arr = np.array(mol_im)
	mol_im_ax = fig.add_subplot(gs[0])
	mol_im_ax.imshow(mol_im_arr)
	mol_im_ax.axis("off")
	# mol_im_ax.patch.set_edgecolor('black')
	# mol_im_ax.patch.set_linewidth('1')

	ax_top = fig.add_subplot(gs[1], facecolor="white")

	bar_top = ax_top.bar(
		x_array,
		true_spec,
		bar_width,
		color="tab:blue",
		edgecolor="tab:blue",
		zorder=0.5
	)

	ax_top.set_ylim(0,y_max)
	ax_top.set_xlim(0,x_max)

	plt.setp(ax_top.get_xticklabels(), visible=False)
	ax_top.grid(color="black", linewidth=0.1)

	ax_bottom = fig.add_subplot(gs[2], facecolor="white")
	fig.subplots_adjust(hspace=0.0)

	bar_bottom = ax_bottom.bar(
		x_array,
		gen_spec,
		bar_width,
		color="tab:red",
		edgecolor="tab:red",
		zorder=0.5
	)

	# Invert the direction of y-axis ticks for bottom graph.
	ax_bottom.set_ylim(y_max, 0)
	ax_bottom.set_xlim(0, x_max)

	def _plot_mbs(mbs,color,ax):
		mb_array = np.zeros_like(x_array,dtype=float)
		# mass bounds are binned
		for bin_l,bin_u in mbs:
			if bin_l > x_array.shape[0]-1:
				# rectangle completely out of range
				continue
			elif bin_u > x_array.shape[0]-1:
				# only upper out of range, set it to max val
				bin_u = x_array.shape[0]-1
			for idx in range(bin_l,bin_u):
				mb_array[idx] = ints_max
			# x_l = x_array[bin_l]
			# x_u = x_array[bin_u]
			# rect = patches.Rectangle((x_l,0),x_u-x_l,y_max,edgecolor=None,facecolor=color,alpha=0.33,zorder=0)
			# ax.add_patch(rect)
		ax.bar(
			x_array,
			mb_array,
			color=color,
			edgecolor=None,
			alpha=0.33,
			zorder=0.
		)

	if not (cfm_mbs is None):
		_plot_mbs(cfm_mbs,"purple",ax_bottom)

	if not (simple_mbs is None):
		_plot_mbs(simple_mbs,"green",ax_top)

	if not (prec_mz_bin is None):
		if prec_mz_bin < x_array.shape[0]: 
			ax_top.axvline(x=prec_mz_bin,color="black",linestyle="dashed",linewidth=bar_width)
			ax_bottom.axvline(x=prec_mz_bin,color="black",linestyle="dashed",linewidth=bar_width)

	# Remove overlapping 0's from middle of y-axis
	yticks_bottom = ax_bottom.yaxis.get_major_ticks()
	yticks_bottom[0].label1.set_visible(False)

	ax_bottom.grid(color="black", linewidth=0.1)

	for ax in [ax_top, ax_bottom]:
		ax.minorticks_on()
		ax.tick_params(axis='y', which='minor', left=False)
		ax.tick_params(axis='y', which='minor', right=False)
		ax.tick_params(axis="both", which="major", labelsize=tick_size)

	ax_top.tick_params(axis='x', which='minor', top=False)

	legend_kws = {'ncol': 1, 'fontsize': font_size, "loc": "upper left"}
	ax_top.legend((bar_top, bar_bottom),
		("Real",
		 "Predicted"),
		**legend_kws
	)
	# tight_layout does NOT work here!
	data = fig_to_data_2(fig,bbox_inches="tight")
	plt.close("all")
	return data

def plot_progression(ces,specs,real_idx,min_ints_frac,mz_res,real_color="tab:blue",pred_color="tab:red",size=36):
	
	ints_max = max(np.max(spec) for spec in specs)
	min_ints = ints_max*min_ints_frac
	for spec in specs:
		spec[spec<min_ints] = 0.
	mz_max = max(np.flatnonzero(spec)[-1]*mz_res for spec in specs)
	x_max = 1.05*mz_max
	y_max = 1.0*ints_max
	bar_width = 0.8*mz_res
	x_array = np.arange(0.,x_max,step=mz_res)
	specs = [spec.flatten()[:x_array.shape[0]] for spec in specs]
	fig = plt.figure(figsize=(20,10), dpi=200)

	# Adding extra subplot so both plots have common x-axis and y-axis labels
	ax_main = fig.add_subplot(111, frameon=False)
	x_pad = int(size)
	y_pad = int(0.5*size)
	tick_size = int(0.8*size)
	ax_main.tick_params(axis="both", which="major", labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax_main.set_xlabel('Mass/Charge (m/z)',fontsize=size,labelpad=x_pad)
	ax_main.set_ylabel('Relative Intensity',fontsize=size,labelpad=y_pad)

	assert len(specs) == 4
	for i in range(len(specs)):
		if i == real_idx:
			title = f"CE = {ces[i]}"
			color = real_color
		else:
			title = f"CE = {ces[i]}"
			color = pred_color
		ax = fig.add_subplot(2,2,(i+1),facecolor="white")
		ax.bar(
			x_array,
			specs[i],
			bar_width,
			color=color,
			edgecolor=color,
			zorder=0.5
		)
		ax.set_xlim(0,x_max)
		ax.set_ylim(0,y_max)
		ax.grid(color="black", linewidth=0.1)
		ax.set_title(title,fontsize=size)
#         ax.xaxis.set_ticks([])
		ax.yaxis.set_ticks([])
#         ax.tick_params(axis="both", which="major", labelcolor='none', top=False, bottom=False, left=False, right=False)
		ax.minorticks_on()
#         ax.tick_params(axis='y', which='minor', left=False)
#         ax.tick_params(axis='y', which='minor', right=False)
		ax.tick_params(axis="both", which="major", labelsize=tick_size)
		ax.tick_params(axis='x', which='minor', top=False)
	fig.tight_layout()
	data = fig_to_data_2(fig)
	plt.close("all")
	return data

def plot_rank_hist(query_ranks,num_candidates):

	fig, ax = plt.subplots(1,1)
	bins = np.arange(0,num_candidates+1)
	ax.hist(query_ranks,bins=bins)
	ax.axvline(x=np.mean(query_ranks),color="black",linestyle="dashed",linewidth=1)
	ax.axvline(x=np.median(query_ranks),color="black",linestyle="solid",linewidth=1)
	ax.set_xlabel("query rank")
	# fig.suptitle("val split: true query ranks")
	data = fig_to_data_2(fig,bbox_inches="tight")
	# plt.savefig("rank_hist.png")
	plt.close("all")
	return data

def plot_num_candidates_hist(num_candidates_per_query):

	fig, ax = plt.subplots(1,1)
	bins = np.arange(0,10100,100)
	ax.hist(num_candidates_per_query,bins=bins)
	ax.axvline(x=np.mean(num_candidates_per_query),color="black",linestyle="dashed",linewidth=1)
	ax.axvline(x=np.median(num_candidates_per_query),color="black",linestyle="solid",linewidth=1)
	ax.set_xlabel("num candidates")
	# fig.suptitle(f"{split} split: num candidates per query")
	data = fig_to_data_2(fig,bbox_inches="tight")
	# plt.savefig("num_candidates_hist.png")
	plt.close("all")
	return data

def plot_cand_sim_mean_hist(query_sim_means):

	fig, ax = plt.subplots(1,1)
	bins = np.linspace(0.,1.,101)
	ax.hist(query_sim_means,bins=bins)
	ax.axvline(x=np.mean(query_sim_means),color="black",linestyle="dashed",linewidth=1)
	ax.axvline(x=np.median(query_sim_means),color="black",linestyle="solid",linewidth=1)
	ax.set_xlabel("mean similarity")
	# fig.suptitle(f"{split} split: mean similarity of candidates per query")
	data = fig_to_data_2(fig,bbox_inches="tight")
	# plt.savefig("hist.png")
	plt.close("all")
	return data

def plot_cand_sim_std_hist(query_sim_stds):

	fig, ax = plt.subplots(1,1)
	ax.hist(query_sim_stds,bins=100)
	ax.axvline(x=np.mean(query_sim_stds),color="black",linestyle="dashed",linewidth=1)
	ax.axvline(x=np.median(query_sim_stds),color="black",linestyle="solid",linewidth=1)
	ax.set_xlabel("std similarity")
	# fig.suptitle(f"{split} split: std similarity of candidates per query")
	data = fig_to_data_2(fig,bbox_inches="tight")
	# plt.savefig("hist.png")
	plt.close("all")
	return data

def plot_sim_hist(sim_type,sims):

	fig, ax = plt.subplots(1,1)
	bins = np.linspace(0.,1.,101)
	ax.hist(sims,bins=bins)
	ax.axvline(x=np.mean(sims),color="black",linestyle="dashed",linewidth=1)
	ax.axvline(x=np.median(sims),color="black",linestyle="solid",linewidth=1)
	ax.set_xlabel(f"{sim_type} similarity")
	# fig.suptitle(f"{split} split: {sim_type} similarity")
	data = fig_to_data_2(fig,bbox_inches="tight")
	# plt.savefig("hist.png")
	plt.close("all")
	return data	

def attn_to_rgb(attn,cmap="Reds"):
	cm = matplotlib.cm.get_cmap(cmap)
	return cm(attn)[:3]

def viz_attention(smiles,attention_map):
	""" note that we assume that the node order in attn_dict is the same as the order in smiles"""

	# import pdb; pdb.set_trace()
	mol = Chem.MolFromSmiles(smiles)
	d = rdMolDraw2D.MolDraw2DCairo(500, 500)
	atom_idxs, bond_idxs = [], []
	atom_colors, bond_colors = {}, {}
	# import pdb; pdb.set_trace()
	for atom_idx, atom in enumerate(mol.GetAtoms()):
		assert atom_idx == atom.GetIdx(), (atom_idx,atom.GetIdx())
		atom_idxs.append(atom_idx)
		attn = attention_map[atom_idx]
		atom_colors[atom_idx] = attn_to_rgb(attn)
	d.DrawMolecules(
		[mol],
		highlightAtoms=[atom_idxs],
		highlightAtomColors=[atom_colors],
		highlightBonds=[bond_idxs],
		highlightBondColors=[bond_colors],
		legends=[smiles]
	)
	d.FinishDrawing()
	png_buf = d.GetDrawingText()
	im = Image.open(io.BytesIO(png_buf))
	return im

def plot_combined_kde(xs1,ys1,xs2,ys2,cmap="Purples",diff_cmap="coolwarm",xmin=None,xmax=None,ymin=None,ymax=None,xlabel=None,ylabel=None,size=36):
	
	if xmin is None:
		xmin = min(xs1.min(),xs2.min())
	if xmax is None:
		xmax = max(xs1.max(),xs2.max())
	if ymin is None:
		ymin = min(ys1.min(),ys1.min())
	if ymax is None:
		ymax = max(ys1.max(),ys1.max())
	fig = plt.figure(figsize=(15,10),dpi=200)
	ax_main = fig.add_subplot(111,frameon=False)
	x_pad = int(size)
	y_pad = int(size)
	ax_main.tick_params(axis="both", which="major", labelcolor='none', top=False, bottom=False, left=False, right=False)
	ax_main.set_xlabel(xlabel,fontsize=size,labelpad=x_pad)
#     ax_main.set_ylabel(ylabel,fontsize=size)
	X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	positions = np.vstack([X.ravel(), Y.ravel()])
	values1 = np.vstack([xs1, ys1])
	kernel1 = gaussian_kde(values1)
	Z1 = np.reshape(kernel1(positions).T, X.shape)
	values2 = np.vstack([xs2, ys2])
	kernel2 = gaussian_kde(values2)
	Z2 = np.reshape(kernel2(positions).T, X.shape)
	# Z3 = Z1-Z2
	ax1 = fig.add_subplot(121)
	ax1.imshow(np.rot90(Z1),cmap=cmap,extent=[xmin, xmax, ymin, ymax])
	ax1.xaxis.set_tick_params(labelsize=int(0.8*size))
#     ax1.yaxis.set_ticks([])
	ax1.yaxis.set_tick_params(labelsize=int(0.8*size))
#     ax1.tick_params(axis="both", which="major", pad=18)
	ax1.set_ylabel(ylabel,fontsize=size,labelpad=y_pad)
	ax2 = fig.add_subplot(122)
	ax2.imshow(np.rot90(Z2),cmap=cmap,extent=[xmin, xmax, ymin, ymax])
	ax2.xaxis.set_tick_params(labelsize=int(0.8*size))
	ax2.yaxis.set_ticks([])
	#     ax2.yaxis.set_tick_params(labelsize=int(0.8*size))
	# ax3 = fig.add_subplot(133)
	# ax3.imshow(np.rot90(Z3),cmap=diff_cmap,extent=[xmin, xmax, ymin, ymax])
	# ax3.xaxis.set_tick_params(labelsize=int(0.8*size))
	# ax3.yaxis.set_ticks([])
	# ax3.yaxis.set_tick_params(labelsize=int(0.8*size))
	fig.tight_layout()
	data = fig_to_data_2(fig)
	plt.close("all")
	return data

def best_fit(x,y):
	xbar = np.mean(x)
	ybar = np.mean(y)
	n = len(x) # or len(y)
	numer = np.sum(x*y) - n * xbar * ybar
	denum = np.sum(x**2) - n * xbar**2
	b = numer / denum
	a = ybar - b * xbar
	print('best fit line:\ny = {:.2f} + {:.2f}x'.format(float(a), float(b)))
	y_bf = a + b*x
	return y_bf

def attn_to_rgb(attn,cmap="Greys"):
	cm = matplotlib.cm.get_cmap(cmap)
	return cm(attn)[:3]

def viz_attention(smiles,attention_map,cmap):
	""" note that we assume that the node order in attn_dict is the same as the order in smiles"""

	# import pdb; pdb.set_trace()
	mol = Chem.MolFromSmiles(smiles)
	d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
	d.drawOptions().useBWAtomPalette()
	atom_idxs, bond_idxs = [], []
	atom_colors, bond_colors = {}, {}
	for atom_idx, atom in enumerate(mol.GetAtoms()):
		assert atom_idx == atom.GetIdx(), (atom_idx,atom.GetIdx())
		atom_idxs.append(atom_idx)
		attn = attention_map[atom_idx]
		atom_colors[atom_idx] = attn_to_rgb(attn,cmap=cmap)
	d.DrawMolecules(
		[mol],
		highlightAtoms=[atom_idxs],
		highlightAtomColors=[atom_colors],
		highlightBonds=[bond_idxs],
		highlightBondColors=[bond_colors],
#         legends=[smiles]
	)
	d.FinishDrawing()
	png_buf = d.GetDrawingText()
	im = Image.open(io.BytesIO(png_buf))
	im = trim_img_by_white(im,padding=15)
	return im

def plot_atom_attn(atom_avg_d,size=36):

	sns_colors = sns.color_palette(n_colors=6)
	size = 36
	font_size = size
	tick_size = int(0.8*size)
	x_pad = int(size)
	y_pad = int(size)
	fig, ax = plt.subplots(figsize=(15,10),dpi=200)
	x_labels, y_vals, y_errs, colors = [], [], [], []
	color_d = {
		"C": sns_colors[0],
		"N": sns_colors[1],
		"O": sns_colors[2],
		"P": sns_colors[3],
		"S": sns_colors[4],
		"Cl": sns_colors[5]
	}
	for k in sorted(atom_avg_d.keys()):
		if k == "?":
			continue
		else:
			x_labels.append(k)
			y_vals.append(atom_avg_d[k][0])
			y_errs.append(atom_avg_d[k][1])
			colors.append(color_d[k])
	x_pos = list(range(1,len(x_labels)+1))
	ax.bar(x_pos, y_vals, yerr=y_errs, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)
	ax.set_xticks(x_pos)
	ax.set_xticklabels(x_labels,fontsize=font_size)
	ax.set_ylabel('Attention Relative to C',fontsize=font_size,labelpad=y_pad)
	ax.set_xlabel('Atom Type',fontsize=font_size,labelpad=x_pad)
	ax.tick_params(axis="both", which="major", labelsize=tick_size)
	ax.yaxis.grid(color="grey")
	ax.set_axisbelow(True)
	fig.tight_layout()
	data = fig_to_data_2(fig)
	return data


def fig_to_data_2(fig,**kwargs):

	buf = io.BytesIO()
	fig.savefig(buf,**kwargs)
	buf.seek(0)
	image = Image.open(buf)
	return image
