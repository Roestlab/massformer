import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image, ImageOps
import io
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
from scipy.stats import gaussian_kde
import seaborn as sns
import cairosvg

from massformer.data_utils import mol_from_smiles
from massformer.misc_utils import EPS


def trim_img_by_white(img, padding=0):
    '''
    This function takes a PIL image, img, and crops it to the minimum rectangle
    based on its whiteness/transparency. 5 pixel padding used automatically.
    Adapted from:
    https://github.com/connorcoley/retrosim/blob/master/retrosim/utils/draw.py
    '''

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)
    assert as_array.ndim == 3 and as_array.shape[2] == 3, as_array.shape
    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)
    # Crop down
    x_range = max([min(xs) - 5, 0]), min([max(xs) + 5, as_array.shape[0]])
    y_range = max([min(ys) - 5, 0]), min([max(ys) + 5, as_array.shape[1]])
    as_array_cropped = as_array[x_range[0]:x_range[1], y_range[0]:y_range[1], 0:3]
    img = Image.fromarray(as_array_cropped, mode='RGB')
    return ImageOps.expand(img, border=padding, fill=(255, 255, 255, 0))


def get_mol_im(smiles):

    width = 1000
    height = 1000
    mols = [mol_from_smiles(smiles)]
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    d.DrawMolecules(mols)
    d.FinishDrawing()
    svg_buf = d.GetDrawingText()
    png_buf = cairosvg.svg2png(svg_buf)
    im = Image.open(io.BytesIO(png_buf))
    im = trim_img_by_white(im, padding=15)
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
        custom_title=None,
        mol_image=None,
        loss_type=None,
        loss=None,
        sim_type=None,
        sim=None,
        height_ratios=None,
        size=24,
        attn_cm=None,
        attn_cm_range=(0., 1.),
        return_as_data=True,
        figsize=(20, 20),
        leg_font_size=None):
    """
    Generates a plot comparing a true and predicted mass spec spectra.
    rescaling is based on the TRUE spec only
    """

    bar_width = 0.8 * mz_res

    if rescale_mz_axis:
        true_mz_max = np.flatnonzero(true_spec)[-1] * mz_res
    x_max = int(np.floor(1.05 * min(mz_max, true_mz_max)))
    x_array = np.arange(0., x_max, step=mz_res)
    x_len = x_array.shape[0]
    true_spec = true_spec.flatten()
    gen_spec = gen_spec.flatten()
    if true_spec.shape[0] < x_len:
        true_spec = np.concatenate(
            [true_spec, np.zeros([x_len - true_spec.shape[0]])], axis=0)
        gen_spec = np.concatenate(
            [gen_spec, np.zeros([x_len - gen_spec.shape[0]])], axis=0)
    else:
        true_spec = true_spec[:x_len]
        gen_spec = gen_spec[:x_len]

    assert np.max(true_spec) != 0.

    if rescale_ints_axis:
        ints_max = np.max(true_spec)
    else:
        ints_max = 1.
    y_max = 1.05 * ints_max

    fig = plt.figure(figsize=figsize, dpi=200)
    if plot_title:
        title_y = 0.975
        title_font_size = 20
        if custom_title is not None:
            fig.suptitle(custom_title, fontsize=title_font_size, y=title_y)
        else:
            # default title
            assert not (loss_type is None)
            assert not (sim_type is None)
            assert not (loss is None)
            assert not (sim is None)
            fig.suptitle(
                f"{loss_type}_loss = {loss:.4g}, {sim_type}_sim = {sim:.4g}",
                fontsize=title_font_size,
                y=title_y)
        if height_ratios is None:
            height_ratios = [1, 3, 3]
    else:
        if height_ratios is None:
            height_ratios = [1, 2, 2]

    # Adding extra subplot so both plots have common x-axis and y-axis labels
    y_pad = int(2.0 * size)
    x_pad = int(size)
    tick_size = int(0.8 * size)
    font_size = int(size)

    # set up gridspec
    left = 0.12
    right = 0.98
    bottom = 0.10
    if not (smiles is None) or not (mol_image is None):
        top = 0.95
        gs = mpl.gridspec.GridSpec(3, 1, height_ratios=height_ratios)
        gs.update(left=left, right=right, top=top, bottom=bottom, hspace=0.)
        if not (mol_image is None):
            mol_im = mol_image
            if not (attn_cm is None):
                attn_colorbar = True
        else:
            mol_im = get_mol_im(smiles)
            attn_colorbar = False  # no attention values
        mol_im_arr = np.array(mol_im)
        mol_im_ax = fig.add_subplot(gs[0])
        mol_cim = mol_im_ax.imshow(mol_im_arr)
        mol_im_ax.axis("off")
        if attn_colorbar:
            cax = fig.add_axes((0.85,
                                sum(height_ratios[1:]) / sum(height_ratios),
                                0.02,
                                height_ratios[0] / sum(height_ratios)))
            iax = inset_axes(
                cax,
                width="100%",
                height="50%",
                loc="center right",
                borderpad=0)
            cax.set_visible(False)
            norm = mpl.colors.Normalize(
                vmin=attn_cm_range[0], vmax=attn_cm_range[1])
            cmap = mpl.cm.get_cmap(attn_cm)
            cb = mpl.colorbar.ColorbarBase(
                iax, cmap=cmap, norm=norm, alpha=0.7, orientation="vertical")
            cb.ax.tick_params(labelsize=0.7 * tick_size)
        ylabel_y = bottom + \
            (height_ratios[-1] / sum(height_ratios)) * (top - bottom)
    else:
        top = 0.98
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=height_ratios[1:])
        gs.update(left=left, right=right, top=top, bottom=bottom, hspace=0.)
        mol_im_ax = None
        ylabel_y = bottom + 0.5 * (top - bottom)

    ax_top = fig.add_subplot(gs[-2], facecolor="white")

    bar_top = ax_top.bar(
        x_array,
        true_spec,
        bar_width,
        color="tab:blue",
        edgecolor="tab:blue",
        zorder=0.5
    )

    ax_top.set_ylim(0, y_max)
    ax_top.set_xlim(0, x_max)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.grid(color="black", linewidth=0.1)

    ax_bottom = fig.add_subplot(gs[-1], facecolor="white")
    # fig.subplots_adjust(hspace=0.0)

    # x/y axis labels
    fig.supxlabel('Mass/Charge (m/z)', fontsize=font_size,
                  x=left + 0.5 * (right - left), y=0.2 * bottom)
    fig.supylabel(
        'Relative Intensity',
        fontsize=font_size,
        y=ylabel_y,
        x=0.2 * left)

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

    def _plot_mbs(mbs, color, ax):
        mb_array = np.zeros_like(x_array, dtype=float)
        # mass bounds are binned
        for bin_l, bin_u in mbs:
            if bin_l > x_array.shape[0] - 1:
                # rectangle completely out of range
                continue
            elif bin_u > x_array.shape[0] - 1:
                # only upper out of range, set it to max val
                bin_u = x_array.shape[0] - 1
            for idx in range(bin_l, bin_u):
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
        _plot_mbs(cfm_mbs, "purple", ax_bottom)

    if not (simple_mbs is None):
        _plot_mbs(simple_mbs, "green", ax_top)

    if not (prec_mz_bin is None):
        if prec_mz_bin < x_array.shape[0]:
            ax_top.axvline(
                x=prec_mz_bin,
                color="black",
                linestyle="dashed",
                linewidth=bar_width)
            ax_bottom.axvline(
                x=prec_mz_bin,
                color="black",
                linestyle="dashed",
                linewidth=bar_width)

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

    if leg_font_size is None:
        leg_font_size = 0.9 * font_size
    leg_kws = {'ncol': 1, 'fontsize': leg_font_size, "loc": "upper left"}
    leg = ax_top.legend((bar_top, bar_bottom),
                        ("Real",
                         "Predicted"),
                        **leg_kws
                        )
    # tight_layout does NOT work here!
    if return_as_data:
        data = fig_to_data(fig)  # ,bbox_inches="tight")
        plt.close("all")
        return data
    else:
        # you need to close it yourself
        mpl_d = {
            "fig": fig,
            "ax_top": ax_top,
            "ax_bottom": ax_bottom,
            "mol_im_ax": mol_im_ax,
            "bar_top": bar_top,
            "bar_bottom": bar_bottom,
            "leg": leg
        }
        return mpl_d


def plot_progression(
    ces,
    real_specs,
    pred_specs,
    mz_res,
    real_color="tab:blue",
    pred_color="tab:red",
    size=28,
    mz_max=None,
    ints_max=None,
    dims=[
        1,
        4]):

    if mz_max is None:
        x_max_idx = max(np.flatnonzero(spec)[-1] for spec in real_specs) + 1
        mz_max = x_max_idx * mz_res
    else:
        x_max_idx = int(np.ceil(mz_max / mz_res)) + 1
    x_max = int(1.05 * mz_max)
    bar_width = 0.8 * mz_res
    x_array = np.arange(0., x_max, step=mz_res)
    assert x_array.shape[0] >= x_max_idx
    # reshape and normalize spectra
    assert len(real_specs) == len(pred_specs)
    viz_real_specs, viz_pred_specs = [], []
    for i in range(len(real_specs)):
        real_spec = real_specs[i].flatten()
        assert np.sum(real_spec[x_max_idx:]) == 0.
        real_spec = real_spec[:x_max_idx] / \
            max(np.sum(real_spec[:x_max_idx]), EPS)
        assert np.isclose(
            np.sum(real_spec),
            1.) or np.isclose(
            np.sum(real_spec),
            0.)
        viz_real_specs.append(np.concatenate(
            [real_spec, np.zeros([x_array.shape[0] - x_max_idx])]))
        pred_spec = pred_specs[i].flatten()
        pred_spec = pred_spec[:x_max_idx] / \
            max(np.sum(pred_spec[:x_max_idx]), EPS)
        assert np.isclose(
            np.sum(pred_spec),
            1.) or np.isclose(
            np.sum(pred_spec),
            0.)
        viz_pred_specs.append(np.concatenate(
            [pred_spec, np.zeros([x_array.shape[0] - x_max_idx])]))
    if ints_max is None:
        ints_max = max(np.max(spec) for spec in viz_real_specs)
    y_max = 1.10 * ints_max

    if dims == [2, 2]:
        fig = plt.figure(figsize=(20, 15), dpi=300)
        x_pad = int(0.3 * size)
        y_pad = int(0.3 * size)
        wspace = 0.15
        hspace = 0.2
    elif dims == [1, 4]:
        fig = plt.figure(figsize=(4 * 8, 6), dpi=300)
        x_pad = int(0.2 * size)
        y_pad = int(0.2 * size)
        wspace = 0.15
        hspace = 0.0
    elif dims == [4, 1]:
        fig = plt.figure(figsize=(7, 4 * 6), dpi=300)
        x_pad = int(0.2 * size)
        y_pad = int(0.2 * size)
        wspace = 0.0
        hspace = 0.2
    elif dims == [3, 1]:
        viz_real_specs = [
            viz_real_specs[0],
            viz_real_specs[1],
            viz_real_specs[2]]
        viz_pred_specs = [
            viz_pred_specs[0],
            viz_pred_specs[1],
            viz_pred_specs[2]]
        fig = plt.figure(figsize=(7, 3 * 6), dpi=300)
        x_pad = int(0.2 * size)
        y_pad = int(0.2 * size)
        wspace = 0.0
        hspace = 0.2
    else:
        raise ValueError(f"dims={dims}")

    # Adding extra subplot so both plots have common x-axis and y-axis labels
    ax_main = fig.add_subplot(111, frameon=False)
    tick_size = int(0.8 * size)
    ax_main.tick_params(
        axis="both",
        which="major",
        labelcolor='none',
        top=False,
        bottom=False,
        left=False,
        right=False)
    ax_main.set_xlabel('Mass/Charge (m/z)', fontsize=size, labelpad=x_pad)
    ax_main.set_ylabel('Relative Intensity', fontsize=size, labelpad=y_pad)

    for i, (real_spec, pred_spec) in enumerate(
            zip(viz_real_specs, viz_pred_specs)):
        title = f"NCE = {ces[i]}%"
        # idk if this affects the plot
        subfig = plt.figure(figsize=(8, 6), dpi=300)
        real_ax = subfig.add_subplot(2, 1, 1, facecolor="white")
        real_ax.bar(
            x_array,
            real_spec,
            bar_width,
            color=real_color,
            edgecolor=real_color,
            zorder=0.5
        )
        real_ax.set_xlim(0, x_max)
        real_ax.set_ylim(0, y_max)
        real_ax.grid(color="black", linewidth=0.1)
        real_ax.yaxis.set_ticks([])
        real_ax.minorticks_on()
        plt.setp(real_ax.get_xticklabels(), visible=False)
        pred_ax = subfig.add_subplot(2, 1, 2, facecolor="white")
        pred_ax.bar(
            x_array,
            pred_spec,
            bar_width,
            color=pred_color,
            edgecolor=pred_color,
            zorder=0.5
        )
        pred_ax.set_xlim(0, x_max)
        pred_ax.set_ylim(y_max, 0)
        pred_ax.grid(color="black", linewidth=0.1)
        pred_ax.yaxis.set_ticks([])
        pred_ax.minorticks_on()
        plt.setp(pred_ax.get_xticklabels(), visible=False)
        subfig.subplots_adjust(hspace=0.0)

        # convert to image and add to main figure
        subfig_data = fig_to_data(subfig, bbox_inches="tight")
        plt.close(subfig)
        subfig_ax = fig.add_subplot(dims[0], dims[1], i + 1, facecolor="white")
        subfig_ax.imshow(subfig_data)  # ,aspect="equal")
        subfig_ax.set_title(title, fontsize=size, pad=0)
        subfig_ax.autoscale(tight=True)
        subfig_ax.set_axis_off()
        # subfig_ax.margins(x=0,y=0,tight=True)

    if dims == [1, 4]:
        real_patch = patches.Patch(color="blue", label="Real")
        predicted_patch = patches.Patch(color="red", label="Predicted")
        ax_main.legend(
            handles=[real_patch, predicted_patch],
            loc=(0.7, -0.18),  # "lower right",
            fontsize=size,
            ncol=2,
            framealpha=1.0
        )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.tight_layout()
    data = fig_to_data(fig, bbox_inches="tight")
    plt.close("all")
    return data


def plot_rank_hist(query_ranks, num_candidates):

    fig, ax = plt.subplots(1, 1)
    bins = np.arange(0, num_candidates + 1)
    ax.hist(query_ranks, bins=bins)
    ax.axvline(
        x=np.mean(query_ranks),
        color="black",
        linestyle="dashed",
        linewidth=1)
    ax.axvline(
        x=np.median(query_ranks),
        color="black",
        linestyle="solid",
        linewidth=1)
    ax.set_xlabel("query rank")
    # fig.suptitle("val split: true query ranks")
    data = fig_to_data(fig, bbox_inches="tight")
    # plt.savefig("rank_hist.png")
    plt.close("all")
    return data


def plot_num_candidates_hist(num_candidates_per_query):

    fig, ax = plt.subplots(1, 1)
    bins = np.arange(0, 10100, 100)
    ax.hist(num_candidates_per_query, bins=bins)
    ax.axvline(
        x=np.mean(num_candidates_per_query),
        color="black",
        linestyle="dashed",
        linewidth=1)
    ax.axvline(
        x=np.median(num_candidates_per_query),
        color="black",
        linestyle="solid",
        linewidth=1)
    ax.set_xlabel("num candidates")
    # fig.suptitle(f"{split} split: num candidates per query")
    data = fig_to_data(fig, bbox_inches="tight")
    # plt.savefig("num_candidates_hist.png")
    plt.close("all")
    return data


def plot_cand_sim_mean_hist(query_sim_means):

    fig, ax = plt.subplots(1, 1)
    bins = np.linspace(0., 1., 101)
    ax.hist(query_sim_means, bins=bins)
    ax.axvline(
        x=np.mean(query_sim_means),
        color="black",
        linestyle="dashed",
        linewidth=1)
    ax.axvline(
        x=np.median(query_sim_means),
        color="black",
        linestyle="solid",
        linewidth=1)
    ax.set_xlabel("mean similarity")
    # fig.suptitle(f"{split} split: mean similarity of candidates per query")
    data = fig_to_data(fig, bbox_inches="tight")
    # plt.savefig("hist.png")
    plt.close("all")
    return data


def plot_cand_sim_std_hist(query_sim_stds):

    fig, ax = plt.subplots(1, 1)
    ax.hist(query_sim_stds, bins=100)
    ax.axvline(
        x=np.mean(query_sim_stds),
        color="black",
        linestyle="dashed",
        linewidth=1)
    ax.axvline(
        x=np.median(query_sim_stds),
        color="black",
        linestyle="solid",
        linewidth=1)
    ax.set_xlabel("std similarity")
    # fig.suptitle(f"{split} split: std similarity of candidates per query")
    data = fig_to_data(fig, bbox_inches="tight")
    # plt.savefig("hist.png")
    plt.close("all")
    return data


def plot_sim_hist(sim_type, sims):

    fig, ax = plt.subplots(1, 1)
    bins = np.linspace(0., 1., 101)
    ax.hist(sims, bins=bins)
    ax.axvline(x=np.mean(sims), color="black", linestyle="dashed", linewidth=1)
    ax.axvline(
        x=np.median(sims),
        color="black",
        linestyle="solid",
        linewidth=1)
    ax.set_xlabel(f"{sim_type} similarity")
    # fig.suptitle(f"{split} split: {sim_type} similarity")
    data = fig_to_data(fig, bbox_inches="tight")
    # plt.savefig("hist.png")
    plt.close("all")
    return data


def plot_combined_kde(
        xs1,
        ys1,
        xs2,
        ys2,
        cmap="Purples",
        diff_cmap="coolwarm",
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        xlabel=None,
        ylabel=None,
        title1=None,
        title2=None,
        size=36):

    if xmin is None:
        xmin = min(xs1.min(), xs2.min())
    if xmax is None:
        xmax = max(xs1.max(), xs2.max())
    if ymin is None:
        ymin = min(ys1.min(), ys1.min())
    if ymax is None:
        ymax = max(ys1.max(), ys1.max())
    fig = plt.figure(figsize=(15, 10), dpi=200)
    ax_main = fig.add_subplot(111, frameon=False)
    x_pad = int(size)
    y_pad = int(size)
    ax_main.tick_params(
        axis="both",
        which="major",
        labelcolor='none',
        top=False,
        bottom=False,
        left=False,
        right=False)
    ax_main.set_xlabel(xlabel, fontsize=size, labelpad=x_pad)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values1 = np.vstack([xs1, ys1])
    kernel1 = gaussian_kde(values1)
    Z1 = np.reshape(kernel1(positions).T, X.shape)
    values2 = np.vstack([xs2, ys2])
    kernel2 = gaussian_kde(values2)
    Z2 = np.reshape(kernel2(positions).T, X.shape)
    ax1 = fig.add_subplot(121)
    ax1.imshow(np.rot90(Z1), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    ax1.xaxis.set_tick_params(labelsize=int(0.8 * size))
    ax1.yaxis.set_tick_params(labelsize=int(0.8 * size))
    ax1.set_ylabel(ylabel, fontsize=size, labelpad=y_pad)
    if not (title1 is None):
        ax1.set_title(title1, fontsize=size, pad=int(0.5 * x_pad))
    ax2 = fig.add_subplot(122)
    ax2.imshow(np.rot90(Z2), cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    ax2.xaxis.set_tick_params(labelsize=int(0.8 * size))
    ax2.yaxis.set_ticks([])
    if not (title2 is None):
        ax2.set_title(title2, fontsize=size, pad=int(0.5 * x_pad))
    fig.tight_layout()
    data = fig_to_data(fig, bbox_inches="tight")
    plt.close("all")
    return data


def best_fit(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    n = len(x)  # or len(y)
    numer = np.sum(x * y) - n * xbar * ybar
    denum = np.sum(x**2) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(float(a), float(b)))
    y_bf = a + b * x
    return y_bf


def attn_to_rgba(attn, cmap="viridis", alpha=0.7):
    cm = mpl.cm.get_cmap(cmap)
    rgb = cm(attn)[:3]
    alpha = alpha * (1. - attn) + 1.0 * attn
    rgba = rgb + (alpha,)
    return rgba


def viz_attention(smiles, attention_map, cmap, alpha):
    """ note that we assume that the node order in attn_dict is the same as the order in smiles"""

    mol = mol_from_smiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d.drawOptions().useBWAtomPalette()
    atom_idxs, bond_idxs = [], []
    atom_colors, bond_colors = {}, {}
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        assert atom_idx == atom.GetIdx(), (atom_idx, atom.GetIdx())
        atom_idxs.append(atom_idx)
        attn = attention_map[atom_idx]
        atom_colors[atom_idx] = attn_to_rgba(attn, cmap=cmap, alpha=alpha)
    d.DrawMolecules(
        [mol],
        highlightAtoms=[atom_idxs],
        highlightAtomColors=[atom_colors],
        highlightBonds=[bond_idxs],
        highlightBondColors=[bond_colors]
    )
    d.FinishDrawing()
    png_buf = d.GetDrawingText()
    im = Image.open(io.BytesIO(png_buf))
    im = trim_img_by_white(im, padding=15)
    return im


def plot_atom_attn(atom_avg_d, size=36):

    sns_colors = sns.color_palette(n_colors=7)
    size = 36
    font_size = size
    tick_size = int(0.8 * size)
    x_pad = int(size)
    y_pad = int(size)
    fig, ax = plt.subplots(figsize=(15, 10), dpi=200)
    x_labels, y_vals, y_errs, colors = [], [], [], []
    color_d = {
        "C": sns_colors[0],
        "N": sns_colors[1],
        "O": sns_colors[2],
        "P": sns_colors[3],
        "S": sns_colors[4],
        "Cl": sns_colors[5],
        "F": sns_colors[6]
    }
    for k in sorted(atom_avg_d.keys()):
        if k == "?":
            continue
        else:
            x_labels.append(k)
            y_vals.append(atom_avg_d[k][0])
            y_errs.append(atom_avg_d[k][1])
            colors.append(color_d[k])
    x_pos = list(range(1, len(x_labels) + 1))
    ax.bar(
        x_pos,
        y_vals,
        yerr=y_errs,
        align='center',
        alpha=0.7,
        ecolor='black',
        capsize=10,
        color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=font_size)
    ax.set_ylabel(
        'Attention Relative to C',
        fontsize=font_size,
        labelpad=y_pad)
    ax.set_xlabel('Atom Type', fontsize=font_size, labelpad=x_pad)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.yaxis.grid(color="grey")
    ax.set_axisbelow(True)
    fig.tight_layout()
    data = fig_to_data(fig)
    return data


def plot_spec_vs_mol_sims(spec_sims, mol_sims, size=0.4):
    fig, ax = plt.subplots()
    ax.scatter(mol_sims, spec_sims, s=size)
    ax.set_ylabel("Prediction target sim (cosine)")
    ax.set_xlabel("Maximum train+val molecule sim (morgan/jaccard)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    data = fig_to_data(fig, bbox_inches="tight")
    plt.close("all")
    return data


def fig_to_data(fig, **kwargs):

    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    image = Image.open(buf)
    return image
