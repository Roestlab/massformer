import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


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
    return fig


def plot_legends():

    # plot_spec, two columns
    real_patch = mpatches.Patch(color="tab:blue",label='Real')
    pred_patch = mpatches.Patch(color="tab:red",label='Predicted')
    patches = [real_patch,pred_patch]
    fig = get_legend(patches, 2)
    fig.savefig("figs/legends/plot_spec_2col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/plot_spec_2col.pdf",bbox_inches="tight",format="pdf")
    plt.close("all")

    # plot_spec, one column
    real_patch = mpatches.Patch(color="tab:blue",label='Real')
    pred_patch = mpatches.Patch(color="tab:red",label='Predicted')
    patches = [real_patch,pred_patch]
    fig = get_legend(patches, 1)
    fig.savefig("figs/legends/plot_spec_1col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/plot_spec_1col.pdf",bbox_inches="tight",format="pdf")
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
    fig = get_legend(patches, 4)
    fig.savefig("figs/legends/models_4col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/models_4col.pdf",bbox_inches="tight",format="pdf")
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
    fig = get_legend(patches, 1)
    fig.savefig("figs/legends/models_1col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/models_1col.pdf",bbox_inches="tight",format="pdf")
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
    fig = get_legend(patches, 3)
    fig.savefig("figs/legends/models_3col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/models_3col.pdf",bbox_inches="tight",format="pdf")
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
    fig = get_legend(patches, 3)
    fig.savefig("figs/legends/classyfire_3col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/classyfire_3col.pdf",bbox_inches="tight",format="pdf")
    plt.close("all")

    # heteroatom separability, two columns
    het_patch = mpatches.Patch(color="green",label="Heteroatom Labelling")
    rand_patch = mpatches.Patch(color="sienna",label="Random Labelling")
    patches = [het_patch,rand_patch]
    fig = get_legend(patches, 2)
    fig.savefig("figs/legends/het_sep_2col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/het_sep_2col.pdf",bbox_inches="tight",format="pdf")
    plt.close("all")

    # heteroatom separability, two columns
    het_patch = mpatches.Patch(color="green",label="Nitrogen Labelling")
    rand_patch = mpatches.Patch(color="sienna",label="Random Labelling")
    patches = [het_patch,rand_patch]
    fig = get_legend(patches, 2)
    fig.savefig("figs/legends/het_sep_n_2col.png",bbox_inches="tight",format="png",dpi=300)
    fig.savefig("figs/legends/het_sep_n_2col.pdf",bbox_inches="tight",format="pdf")
    plt.close("all")

if __name__ == "__main__":

    plot_legends()