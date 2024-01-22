from pprint import pprint
import pandas as pd
import numpy as np
import itertools
import numpy as np
import scipy


if __name__ == "__main__":

    # assumes you have already run scripts/compute_test_metrics.py
    df = pd.read_csv("figs/test_metrics/all_table.csv")
    rank_df_rows = []

    # compute rankings
    dsets_splits_prectypes = df[["dset","split","prec_type"]].drop_duplicates().values.tolist()
    models = df["model"].unique().tolist()
    sim_funcs = df["sim_func"].unique().tolist()
    transforms = df["transform"].unique().tolist()
    prec_peaks = df["prec_peak"].unique().tolist()
    merges = df["merge"].unique().tolist()
    aggregations = df["aggregation"].unique().tolist()
    for dset_split_prectype in dsets_splits_prectypes:
        print(f">>> Starting new dataset/split/prec_type {dset_split_prectype}")
        scenarios_iter = itertools.product(
            sim_funcs,
            transforms,
            prec_peaks,
            merges,
            aggregations
        )
        scenarios = []
        for scenario in scenarios_iter:
            sim_func = scenario[0]
            transform = scenario[1]
            if sim_func == "jacc" and transform != "none":
                continue
            scenarios.append(scenario)
        # print(len(scenarios))
        df_ranks = []
        all_ranks = []
        d_ref = {
            "dset": dset_split_prectype[0],
            "split": dset_split_prectype[1],
            "prec_type": dset_split_prectype[2],
            "sim_func": "cos",
            "transform": "none",
            "prec_peak": "keep",
            "merge": "merge",
            "aggregation": "mol"
        }
        ref_ranks = None
        mf_idx = sorted(models).index("MF")
        for scenario in scenarios:
            d_scenario = {
                "dset": dset_split_prectype[0],
                "split": dset_split_prectype[1],
                "prec_type": dset_split_prectype[2],
                "sim_func": scenario[0],
                "transform": scenario[1],
                "prec_peak": scenario[2],
                "merge": scenario[3],
                "aggregation": scenario[4]
            }
            df_mask = np.all(np.vstack([(df[k] == v).to_numpy() for k,v in d_scenario.items()]).T,axis=1)
            df_scenario = df[df_mask]
            assert df_scenario.shape[0] == len(models), df_scenario.shape
            # if df_scenario.shape[0] != len(models):
            #     import pdb; pdb.set_trace()
            df_scenario = df_scenario.sort_values(by="mean",ascending=False)
            df_scenario["rank"] = np.arange(df_scenario.shape[0])+1
            df_scenario = df_scenario.sort_values(by="model",ascending=True)
            df_ranks.append(df_scenario)
            all_ranks.append(df_scenario["rank"].values)
            if all(d_scenario[k] == d_ref[k] for k in d_ref.keys()):
                assert ref_ranks is None
                ref_ranks = df_scenario["rank"].values
        assert ref_ranks is not None
        df_ranks = pd.concat(df_ranks)
        # compute ranking correlations
        pairwise_rank_correlations = []
        for (ranks_1, ranks_2) in itertools.combinations(all_ranks,2):
            rank_correlation = scipy.stats.pearsonr(ranks_1,ranks_2).statistic
            pairwise_rank_correlations.append(rank_correlation)
        pairwise_rank_correlations = np.array(pairwise_rank_correlations)
        # print(f"> Mean pairwise rank correlation = {np.mean(pairwise_rank_correlations)}")
        ref_rank_correlations, ref_rank_identicals, mf_bests = [], [], []
        for ranks in all_ranks:
            ref_rank_correlation = scipy.stats.pearsonr(ref_ranks,ranks).statistic
            ref_rank_correlations.append(ref_rank_correlation)
            ref_rank_identical = np.all(np.array(ref_ranks) == np.array(ranks))
            ref_rank_identicals.append(ref_rank_identical)
            mf_best = ranks[mf_idx] == 1
            mf_bests.append(mf_best)
        ref_rank_correlations = np.array(ref_rank_correlations)
        ref_rank_identicals = np.array(ref_rank_identicals, dtype=np.float32)
        mf_bests = np.array(mf_bests, dtype=np.float32)

        rank_df_row = {
            "dset": dset_split_prectype[0],
            "split": dset_split_prectype[1],
            "prec_type": dset_split_prectype[2],
            "mean_pairwise_rank_correlation": np.mean(pairwise_rank_correlations),
            "mean_ref_rank_correlation": np.mean(ref_rank_correlations),
            "mean_ref_rank_identical": np.mean(ref_rank_identicals),
            "mean_mf_best": np.mean(mf_bests)
        }
        rank_df_rows.append(rank_df_row)

    rank_df = pd.DataFrame(rank_df_rows)
    rank_df = rank_df[["dset","prec_type","split","mean_pairwise_rank_correlation","mean_ref_rank_correlation","mean_ref_rank_identical","mean_mf_best"]]
    print(rank_df)
    rank_df.to_csv("figs/test_metrics/all_table_ranks.csv",index=False)