import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image, ImageOps
import io
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
from scipy.stats import gaussian_kde
import cairosvg

from massformer.data_utils import mol_from_smiles


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


def get_mol_im(smiles,svg=False):

    width = 1000
    height = 1000
    mols = [mol_from_smiles(smiles)]
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    atom_palette = {
        9: (0/255, 169/255, 169/255), # fluorine
        16: (151/255, 151/255, 0/255), # sulfur
        17: (0/255, 128/255, 14/255) # chlorine
    }
    d.drawOptions().updateAtomPalette(atom_palette)
    d.drawOptions().minFontSize = 32
    d.drawOptions().maxFontSize = 64
    if svg:
        d.drawOptions().clearBackground = False
    d.DrawMolecules(mols)
    d.FinishDrawing()
    svg_buf = d.GetDrawingText()
    if svg:
        im = svg_buf.encode()
    else:
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
        rescale_mz_min=400.,
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
        leg_font_size=None,
        mol_image_overlay=False,
        mol_image_areafrac=0.25,
        include_legend=True,
        include_axes_labels=True,
        include_grid=True):
    """
    Generates a plot comparing a true and predicted mass spec spectra.
    rescaling is based on the TRUE spec only
    """

    bar_width = 0.8 * mz_res
    # compute x_max
    true_mz_max = np.flatnonzero(true_spec)[-1] * mz_res
    if np.max(gen_spec) <= 0.:
        gen_mz_max = 0.
    else:
        gen_mz_max = np.flatnonzero(gen_spec)[-1] * mz_res
    if rescale_mz_axis:
        x_max = max(true_mz_max, gen_mz_max)
        x_max = max(int(np.floor(1.05 * min(mz_max, x_max))), rescale_mz_min)
    else:
        x_max = mz_max
    # round up to the nearest 100
    x_max = int(100 * np.ceil(x_max / 100))
    # check x_max bounds
    if x_max < true_mz_max:
        print(f"WARNING: x_max={x_max} < true_mz_max={true_mz_max}")
    if x_max < gen_mz_max:
        print(f"WARNING: x_max={x_max} < gen_mz_max={gen_mz_max}")
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
        ints_max = max(np.max(true_spec), np.max(gen_spec))
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

    # set default values for mol_im
    mol_im, mol_im_svg = None, None

    # set up gridspec
    left = 0.12
    right = 0.98
    bottom = 0.10
    include_image = not (mol_image is None) or not (smiles is None)
    if include_image and not mol_image_overlay:
        top = 0.95
        gs = mpl.gridspec.GridSpec(3, 1, height_ratios=height_ratios)
        gs.update(left=left, right=right, top=top, bottom=bottom, hspace=0.)
        if not (mol_image is None):
            mol_im = mol_image
            mol_im_svg = None
            if not (attn_cm is None):
                attn_colorbar = True
        else:
            mol_im = get_mol_im(smiles)
            mol_im_svg = get_mol_im(smiles,svg=True)
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
    if include_grid:
        ax_top.grid(color="black", linewidth=0.1)
    else:
        ax_top.grid(False)

    ax_bottom = fig.add_subplot(gs[-1], facecolor="white")
    # fig.subplots_adjust(hspace=0.0)

    # x/y axis labels
    if include_axes_labels:
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
            # rect = mpatches.Rectangle((x_l,0),x_u-x_l,y_max,edgecolor=None,facecolor=color,alpha=0.33,zorder=0)
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

    if include_grid:
        ax_bottom.grid(color="black", linewidth=0.1)
    else:
        ax_bottom.grid(False)

    for ax in [ax_top, ax_bottom]:
        ax.minorticks_on()
        ax.tick_params(axis='y', which='minor', left=False)
        ax.tick_params(axis='y', which='minor', right=False)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)

    ax_top.tick_params(axis='x', which='minor', top=False)

    if include_legend:
        if leg_font_size is None:
            leg_font_size = 0.9 * font_size
        leg_kws = {'ncol': 1, 'fontsize': leg_font_size, "loc": "upper left"}
        leg = ax_top.legend((bar_top, bar_bottom),
                            ("Real",
                            "Predicted"),
                            **leg_kws
                            )
    else:
        leg = None
    if include_image and mol_image_overlay:
        # add mol image
        if not (mol_image is None):
            mol_im = mol_image
            mol_im_svg = None
            if not (attn_cm is None):
                attn_colorbar = True
        else:
            mol_im = get_mol_im(smiles)
            mol_im_svg = get_mol_im(smiles,svg=True)
            attn_colorbar = False
        # mol_im.show()
        mol_im_arr = np.array(mol_im)
        # convert to RGBA
        mol_im_arr = np.concatenate([mol_im_arr, np.zeros_like(mol_im_arr[:,:,0:1])], axis=-1)
        mol_im_mask_arr = np.all(mol_im_arr[:,:,:3] == 255, axis=-1)
        mol_im_arr[mol_im_mask_arr,3] = int(0.6*255)
        mol_im_arr[~mol_im_mask_arr,3] = 255
        # aspect ratio
        mol_im_height, mol_im_width = mol_im_arr.shape[:2]
        mol_im_hwratio = mol_im_height / mol_im_width
        # print(mol_im_hwratio)
        ax_top_x_min, ax_top_x_max = ax_top.get_xlim()
        ax_top_y_min, ax_top_y_max = ax_top.get_ylim()
        yx_ratio = (ax_top_y_max - ax_top_y_min) / (ax_top_x_max - ax_top_x_min)
        area = (ax_top_x_max - ax_top_x_min) * (ax_top_y_max - ax_top_y_min)
        # print(ax_top_x_min, ax_top_x_max)
        # print(ax_top_y_min, ax_top_y_max)
        # print(yx_ratio)
        mol_im_hwratio2 = mol_im_hwratio * yx_ratio * (sum(height_ratios[1:]) / height_ratios[1])
        mol_im_hwratio3 = mol_im_hwratio2 * (figsize[0] / figsize[1])
        # print(mol_im_hwratio2)
        mol_im_areafrac = mol_image_areafrac
        mol_im_width = min(
            np.sqrt(mol_im_areafrac * area / mol_im_hwratio3),
            (ax_top_x_max - ax_top_x_min) * 0.50
        )
        mol_im_height = mol_im_width * mol_im_hwratio3
        # print(mol_im_width, mol_im_height, mol_im_height / mol_im_width)
        mol_im_left = 0.98*ax_top_x_max - mol_im_width
        mol_im_right = 0.98*ax_top_x_max
        mol_im_bottom = 0.98*ax_top_y_max - mol_im_height
        mol_im_top = 0.98*ax_top_y_max
        # print(mol_im_left, mol_im_right, mol_im_bottom, mol_im_top)
        ax_top.imshow(
            mol_im_arr,
            aspect="auto",
            extent=[mol_im_left,mol_im_right,mol_im_bottom,mol_im_top],
            zorder=3
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
            "leg": leg,
            "mol_im_png": mol_im,
            "mol_im_svg": mol_im_svg,
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
    mz_min=None,
    mz_max=None,
    ints_max=None,
    dims=[1,4],
    include_legend=True):

    bar_width = 0.8 * mz_res
    # compute x_max
    real_mz_max = max([np.flatnonzero(real_spec)[-1] * mz_res for real_spec in real_specs])
    pred_mz_max = max([np.flatnonzero(pred_spec)[-1] * mz_res for pred_spec in pred_specs])
    if mz_max is None:
        x_max = max(real_mz_max, pred_mz_max)
        x_max = int(np.floor(1.05 * min(mz_max, x_max)))
    else:
        x_max = mz_max
    # round up to the nearest 10
    x_max = int(10 * np.ceil(x_max / 10))
    # check x_max bounds
    if x_max < real_mz_max:
        print(f"WARNING: x_max={x_max} < real_mz_max={real_mz_max}")
    if x_max < pred_mz_max:
        print(f"WARNING: x_max={x_max} < pred_mz_max={pred_mz_max}")
    # compute x_max
    real_mz_min = min([np.flatnonzero(real_spec)[0] * mz_res for real_spec in real_specs])
    pred_mz_min = min([np.flatnonzero(pred_spec)[0] * mz_res for pred_spec in pred_specs])
    if mz_min is None:
        x_min = min(real_mz_min, pred_mz_min)
        x_min = int(np.floor(1.05 * min(mz_min, x_min)))
    else:
        x_min = mz_min
    # round up to the nearest 10
    x_min = int(10 * np.ceil(x_min / 10))
    # print(x_min, x_max)
    # check x_max bounds
    if x_min < real_mz_min:
        print(f"WARNING: x_min={x_min} < real_mz_min={real_mz_min}")
    if x_min < pred_mz_min:
        print(f"WARNING: x_min={x_min} < pred_mz_min={pred_mz_min}")
    # set up x_array
    x_array = np.arange(x_min, x_max, step=mz_res)
    x_min_idx = np.arange(0., x_min, step=mz_res).shape[0]
    x_max_idx = np.arange(0., x_max, step=mz_res).shape[0]
    # reshape and normalize spectra
    assert len(real_specs) == len(pred_specs)
    viz_real_specs, viz_pred_specs = [], []
    for i in range(len(real_specs)):
        real_spec = real_specs[i].flatten()
        if x_max_idx > real_spec.shape[0]:
            real_spec = np.concatenate(
                [real_spec, np.zeros([x_max_idx - real_spec.shape[0]])])
        real_spec = real_spec[x_min_idx:x_max_idx]
        # print(real_spec.shape, x_array.shape)
        assert real_spec.shape[0] == x_array.shape[0]
        viz_real_specs.append(real_spec)
        pred_spec = pred_specs[i].flatten()
        if x_max_idx > pred_spec.shape[0]:
            pred_spec = np.concatenate(
                [pred_spec, np.zeros([x_max_idx - pred_spec.shape[0]])])
        pred_spec = pred_spec[x_min_idx:x_max_idx]
        # print(pred_spec.shape, x_array.shape)
        assert pred_spec.shape[0] == x_array.shape[0]
        viz_pred_specs.append(pred_spec)
    if ints_max is None:
        ints_max = max(np.max(spec) for spec in viz_real_specs)
    y_max = 1.10 * ints_max

    if dims == [2, 2]:
        # fig = plt.figure(figsize=(20, 15), dpi=300)
        # gs = gridspec.GridSpec(2*2, 2)
        # x_pad = int(0.3 * size)
        # y_pad = int(0.3 * size)
        # wspace = 0.15
        # hspace = 0.2
        raise ValueError("dims=[2,2] not supported")
    elif dims == [1, 4]:
        fig = plt.figure(figsize=(4 * 8, 6), dpi=300)
        gs = gridspec.GridSpec(1*2, 4)
        x_pad = int(0.2 * size)
        y_pad = int(0.2 * size)
        wspace = 0.15
        hspace = 0.0
    else:
        raise ValueError(f"dims={dims}")

    # Adding extra subplot so both plots have common x-axis and y-axis labels
    tick_size = int(0.8 * size)
    fig.supxlabel('Mass/Charge (m/z)', fontsize=size, x=0.5, y=0.05)
    fig.supylabel('Relative Intensity', fontsize=size, x=0.05, y=0.5)

    axes = []
    for i, (real_spec, pred_spec) in enumerate(
            zip(viz_real_specs, viz_pred_specs)):
        title = f"NCE = {ces[i]}%"
        real_ax = fig.add_subplot(gs[0,i], facecolor="white")
        pred_ax = fig.add_subplot(gs[1,i], facecolor="white")
        real_ax.bar(
            x_array,
            real_spec,
            bar_width,
            color=real_color,
            edgecolor=real_color,
            zorder=0.5
        )
        real_ax.set_xlim(x_min, x_max)
        real_ax.set_ylim(0, y_max)
        real_ax.grid(color="grey", linewidth=0.01, alpha=0.3)
        real_ax.yaxis.set_ticks([])
        real_ax.set_title(title, fontsize=size, pad=y_pad)
        real_ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.setp(real_ax.get_xticklabels(), visible=False)
        pred_ax.bar(
            x_array,
            pred_spec,
            bar_width,
            color=pred_color,
            edgecolor=pred_color,
            zorder=0.5
        )
        pred_ax.set_xlim(x_min, x_max)
        pred_ax.set_ylim(y_max, 0)
        pred_ax.grid(color="grey", linewidth=0.01, alpha=0.3)
        pred_ax.yaxis.set_ticks([])
        pred_ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.setp(pred_ax.get_xticklabels(), visible=False)
        axes.append((real_ax, pred_ax))

        # # convert to image and add to main figure
        # subfig_data = fig_to_data(subfig, bbox_inches="tight")
        # plt.close(subfig)
        # subfig_ax = fig.add_subplot(dims[0], dims[1], i + 1, facecolor="white")
        # subfig_ax.imshow(subfig_data)  # ,aspect="equal")
        # subfig_ax.set_title(title, fontsize=size, pad=0)
        # subfig_ax.autoscale(tight=True)
        # subfig_ax.set_axis_off()
        # # subfig_ax.margins(x=0,y=0,tight=True)

    if dims == [1, 4] and include_legend:
        real_patch = mpatches.Patch(color="tab:blue", label="Real")
        predicted_patch = mpatches.Patch(color="tab:red", label="Predicted")
        fig.legend(
            handles=[real_patch, predicted_patch],
            loc=(0.7, -0.18),  # "lower right",
            fontsize=size,
            ncol=2,
            framealpha=1.0
        )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.tight_layout()
    # if return_as_data:
    #     data = fig_to_data(fig, bbox_inches="tight")
    #     plt.close("all")
    #     return data
    # else:
    # you need to close it yourself
    mpl_d = {
        "fig": fig,
        "axes": axes,
    }
    return mpl_d


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


def plot_sim_hist(sims, sim_type=None, weights=None, title=None, fp=None, figsize=(9,4), legend=False, size=20, color="blue"):

    big_font_size = int(size)
    small_font_size = int(0.8*size)
    tick_size = int(0.7*size)
    x_pad = y_pad = int(0.6*size)
    fig, ax = plt.subplots(figsize=figsize,dpi=200)
    bins = np.linspace(0., 1., 101)
    if weights is None:
        sims_mean = np.mean(sims)
        weights = np.ones_like(sims,dtype=np.float32) / sims.shape[0]
    else:
        sims_mean = np.average(sims, weights=weights)
    ax.hist(sims, bins=bins, weights=weights, color=color)
    ax.axvline(
        x=sims_mean,
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean={sims_mean:.2f}")
    if sim_type is not None:
        xlabel = f"{sim_type.capitalize()} Similarity"
    else:
        xlabel = "Similarity"
    ax.set_xlabel(
        xlabel,
        fontsize=small_font_size,
        labelpad=x_pad)
    ax.set_ylabel(
        "Fraction",
        fontsize=small_font_size,
        labelpad=y_pad)
    ax.tick_params(axis="x", which="both", labelsize=tick_size)
    ax.tick_params(axis="y", which="both", labelsize=tick_size)
    if title is not None:
        fig.suptitle(title,fontsize=big_font_size)
    if legend:
        ax.legend(
            # handles=handles,
            loc="upper right",
            ncol=1,
            framealpha=1.0,
            fontsize=small_font_size)
    fig.tight_layout()
    data = fig_to_data(fig, bbox_inches="tight")
    if fp is not None:
        plt.savefig(fp)
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
        size=36,
        return_as_data=True):

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
    if return_as_data:
        data = fig_to_data(fig, bbox_inches="tight")
        plt.close("all")
        return data
    else:
        # you need to close it yourself
        mpl_d = {
            "fig": fig,
            "ax_main": ax_main,
            "ax1": ax1,
            "ax2": ax2,
        }
        return mpl_d


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


def viz_attention(smiles, attention_map, cmap, alpha, svg=False):
    """ note that we assume that the node order in attn_dict is the same as the order in smiles"""

    mol = mol_from_smiles(smiles)
    if svg:
        d = rdMolDraw2D.MolDraw2DSVG(1000, 1000)
    else:
        d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d.drawOptions().useBWAtomPalette()
    if svg:
        d.drawOptions().clearBackground = False
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
    buf = d.GetDrawingText()
    # print(buf)
    # import pdb; pdb.set_trace()
    if svg:
        im = buf.encode()
    else:
        im = Image.open(io.BytesIO(buf))
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


def test_plot_spectra():

    from massformer.spec_utils import unprocess_spec
    import torch as th

    mz_max = 1000.
    mz_res = 1.0
    # smiles = "O=C(C)Oc1ccccc1C(=O)O" # aspirin
    # smiles = "COC1C=COC2(C)Oc3c(C)c(O)c4c(O)c(c(C=NN5CCN(C)CC5)c(O)c4c3C2=O)NC(=O)C(C)=CC=CC(C)C(O)C(C)C(O)C(C)C(OC(C)=O)C1C"
    # smiles = "C(C(C(F)(F)Cl)(F)F)(F)F"
    # smiles = "C1=CC(=C(C=C1Cl)S(=O)(=O)O)C(F)(F)F"
    smiles = "C1CN(CCN1)C(C2=CC=CC=C2)C3=CC=CC=C3" # benzhydrylpiperazine
    loss = 1.
    sim = 0.
    loss_type = "cos"
    sim_type = "cos"
    height_ratios = [1,1,1]
    ###
    size = 6
    figsize = (5,4)

    bins = np.arange(0,mz_max+mz_res,step=mz_res)

    # generate random spectrum
    pred_bin_idx = np.random.choice(np.arange(bins.shape[0]),replace=False,size=(10,))
    pred_ints = np.random.rand(10,)
    pred_ints = pred_ints / np.sum(pred_ints)
    pred_spec = np.zeros_like(bins,dtype=np.float32)
    pred_spec[pred_bin_idx] = pred_ints

    # manually define true spectrum
    true_mzs = [
        55.4,
        55.9,
        102.3,
        210.455,
        233.33,
        233.34,
        233.43,
        401.2,
    ]
    true_mzs = np.array(true_mzs)
    true_ints = np.random.rand(*true_mzs.shape)
    true_bin_idx = np.searchsorted(bins,true_mzs,side="right")
    true_spec = np.zeros_like(bins,dtype=np.float32)
    for i in range(len(true_mzs)):
        if true_bin_idx[i] < len(bins):
            true_spec[true_bin_idx[i]] = max(true_spec[true_bin_idx[i]],true_ints[i])

    un_pred_spec = unprocess_spec(th.as_tensor(pred_spec), "none").numpy()
    un_true_spec = unprocess_spec(th.as_tensor(true_spec), "none").numpy()

    plot_d = plot_spec(
        un_true_spec,
        un_pred_spec,
        mz_max,
        mz_res,
        smiles=smiles,
        plot_title=False,
        sim_type=sim_type,
        height_ratios=[1,1,1],
        size=size,
        mol_image_overlay=True,
        rescale_mz_axis=True,
        figsize=figsize,
        include_legend=True,
        include_axes_labels=True,
        mol_image_areafrac=0.25,
        return_as_data=False,
        include_grid=False
    )
    plot_d["fig"].savefig("figs/test_spec_1.png",format="png",dpi=300)
    plot_d["fig"].savefig("figs/test_spec_1.pdf",format="pdf")
    plot_d["mol_im_png"].save("figs/test_mol_1.png",format="png")
    with open("figs/test_mol_1.svg","wb") as f:
        f.write(plot_d["mol_im_svg"])

    mol_im = get_mol_im(smiles)
    mol_im.save("figs/test_mol_1.png")

    save_legends()

def get_legend(patches, ncol):

    fig = plt.figure(figsize=(5,5),dpi=300)
    fig.legend(
        handles=patches,
        loc="center",
        ncol=ncol,
        framealpha=1.0,
        fontsize=30,
        frameon=False)
    fig.tight_layout()
    data = fig_to_data(fig, bbox_inches="tight")
    return data

def save_legends():

    # plot_spec, two columns
    real_patch = mpatches.Patch(color="tab:blue",label='Real')
    pred_patch = mpatches.Patch(color="tab:red",label='Predicted')
    patches = [real_patch,pred_patch]
    data = get_legend(patches, 2)
    data.save("figs/legends/plot_spec_2col.png")
    plt.close("all")

    # plot_spec, one column
    real_patch = mpatches.Patch(color="tab:blue",label='Real')
    pred_patch = mpatches.Patch(color="tab:red",label='Predicted')
    patches = [real_patch,pred_patch]
    data = get_legend(patches, 1)
    data.save("figs/legends/plot_spec_1col.png")
    plt.close("all")

    # model comparisons, four columns
    models = [
        "CFM",
        "FP",
        "WLN",
        "MF"
    ]
    colors = sns.color_palette("muted",n_colors=len(models))
    colors = [
        colors[3],
        colors[0],
        colors[1],
        colors[2]
    ]
    patches = []
    for i, cf_code in enumerate(models):
        patches.append(mpatches.Patch(color=colors[i],label=cf_code))
    data = get_legend(patches, 4)
    data.save("figs/legends/models_4col.png")
    plt.close("all")

    # model comparisons, three columns
    models = [
        "CFM",
        "FP",
        "WLN",
        "MF"
    ]
    colors = sns.color_palette("muted",n_colors=len(models))
    colors = [
        colors[3],
        colors[0],
        colors[1],
        colors[2]
    ]
    patches = []
    for i, cf_code in enumerate(models):
        patches.append(mpatches.Patch(color=colors[i],label=cf_code))
    data = get_legend(patches, 1)
    data.save("figs/legends/models_1col.png")
    plt.close("all")

    # model comparisons, three columns
    models = [
        "FP",
        "WLN",
        "MF"
    ]
    colors = sns.color_palette("muted",n_colors=len(models))
    patches = []
    for i, cf_code in enumerate(models):
        patches.append(mpatches.Patch(color=colors[i],label=cf_code))
    data = get_legend(patches, 3)
    data.save("figs/legends/models_3col.png")
    plt.close("all")

    # classyfire comparisons, three columns
    cf_codes = [
        "Hydrocarbon Derivatives",
        "Organopnictogen Compounds",
        "Benzenoids",
        "Organoheterocyclic Compounds",
        "Phenylpropanoids and Polyketides",
        "Organic Acids and Derivatives",
        "Organic Oxygen Compounds",
        "Lipids and Lipid-like Molecules",
        "Organic Zwitterions",
        "Organic Nitrogen Compounds"
    ]
    colors = sns.color_palette("muted",n_colors=len(cf_codes))
    patches = []
    for i, cf_code in enumerate(cf_codes):
        patches.append(mpatches.Patch(color=colors[i],label=cf_code))
    data = get_legend(patches, 3)
    data.save("figs/legends/classyfire_3col.png")
    plt.close("all")

    # heteroatom separability, two columns
    het_patch = mpatches.Patch(color="green",label="Heteroatom Labelling")
    rand_patch = mpatches.Patch(color="sienna",label="Random Labelling")
    patches = [het_patch,rand_patch]
    data = get_legend(patches, 2)
    data.save("figs/legends/het_sep_2col.png")
    plt.close("all")

    # heteroatom separability, two columns
    het_patch = mpatches.Patch(color="green",label="Nitrogen Labelling")
    rand_patch = mpatches.Patch(color="sienna",label="Random Labelling")
    patches = [het_patch,rand_patch]
    data = get_legend(patches, 2)
    data.save("figs/legends/het_sep_n_2col.png")
    plt.close("all")


if __name__ == "__main__":

    np.random.seed(420)
    test_plot_spectra()
