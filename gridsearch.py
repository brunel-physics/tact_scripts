# -*- coding: utf-8 -*-

"""
mva_analysis.py

Usage:
    tact config.yaml
or  tact --stdin < config.yaml
"""

from __future__ import absolute_import, division, print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tact import binning, classifiers, config, metrics
from tact import plotting as pt
from tact import preprocessing, rootIO


def fit_and_score(estimator, X, y, train, test, sample_weight):

    estimator.fit(X.iloc[train], y.iloc[train],
                  sample_weight=sample_weight.iloc[train])
    roc_test = roc_auc_score(y.iloc[test], estimator.predict_proba(
        X.iloc[test])[:, 1], sample_weight=sample_weight.iloc[test])
    roc_train = roc_auc_score(y.iloc[train], estimator.predict_proba(
        X.iloc[train])[:, 1], sample_weight=sample_weight.iloc[train])

    # print(roc_test)
    # print(roc_train)
    # print(-roc_test + abs(roc_test - roc_train))

    return roc_test - abs(roc_test - roc_train)


def main():
    # Read configuration
    try:
        config.read_config()
    except IndexError:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    cfg = config.cfg

    np.random.seed(777)

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"], cfg["mva_dir"])

    # Read samples
    df = rootIO.read_trees(
        cfg["input_dir"], cfg["features"], cfg["signals"], cfg["backgrounds"],
        selection=cfg["selection"],
        negative_weight_treatment=cfg["negative_weight_treatment"],
        equalise_signal=cfg["equalise_signal"])

    features = cfg["features"]

    # Configure preprocessing
    pre = []
    for p in cfg["preprocessors"]:
        if p["preprocessor"] == "robust_scaler":
            preprocessing.add_robust_scaler(pre, **p["config"])
        elif p["preprocessor"] == "standard_scaler":
            preprocessing.add_standard_scaler(pre, **p["config"])
        elif p["preprocessor"] == "PCA":
            preprocessing.add_PCA(pre, **p["config"])

    # Make plots
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    pt.make_variable_histograms(df[features], df.Signal, w=df.EvtWeight,
                                bins=42, filename="{}vars_{}.pdf"
                                .format(cfg["plot_dir"], cfg["channel"]))
    pt.make_corelation_plot(sig_df[features], w=sig_df.MVAWeight,
                            filename="{}corr_sig_{}.pdf"
                            .format(cfg["plot_dir"], cfg["channel"]))
    pt.make_corelation_plot(bkg_df[features], w=bkg_df.MVAWeight,
                            filename="{}corr_bkg_{}.pdf"
                            .format(cfg["plot_dir"], cfg["channel"]))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=cfg["test_fraction"],
                                         stratify=df.Process)

    # Best features
    from sklearn.feature_selection import mutual_info_classif, f_classif

    # Classify
    from xgboost import XGBClassifier
    # from lightgbm import LGBMClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold

    # from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from sklearn.base import clone
    from skopt import gp_minimize, gbrt_minimize
    from sklearn.externals.joblib import Parallel, delayed

    skf = StratifiedKFold(n_splits=4, shuffle=False)
    bdt = XGBClassifier(silent=True)
    pipe = make_pipeline(*(pre + [bdt]))

    space = [Real(1e-5, 0.1, "log-uniform", name="learning_rate"),
             Integer(32, 5000, name="n_estimators"),
             Integer(2, 8, name="max_depth"),
             Real(0.5, 0.8, name="subsample"),
             Real(1e-5, 1e4, "log-uniform", name="reg_alpha"),
             Real(1e-5, 1e4, "log-uniform", name="reg_lambda"),
             Real(1e-5, 1e3, name="min_child_weight"),
             Real(1e-5, 10, "log-uniform", name="gamma")]

    from sklearn.model_selection import cross_val_score

    parallel = Parallel(n_jobs=4, verbose=True, pre_dispatch=8)

    @use_named_args(space)
    def objective(**params):

        bdt.set_params(n_jobs=4, **params)

        scores = parallel(
            delayed(fit_and_score)(
                clone(bdt), df_train[features], df_train.Signal, train, test, df_train.MVAWeight)
            for train, test in skf.split(df_train, df_train.Process))

        print(scores)

        return -np.mean(scores)

    res_gp = gp_minimize(objective, space, n_calls=100,
                         n_random_starts=10, n_jobs=1, verbose=True, noise=1e-10)

    from skopt.plots import plot_convergence, plot_evaluations, plot_objective

    plot_objective(res_gp)
    plt.savefig("plots/objective_{}.pdf".format(cfg["channel"]))

    plot_evaluations(res_gp)
    plt.savefig("plots/evaluations_{}.pdf".format(cfg["channel"]))

    fig, ax = plt.subplots()
    plot_convergence(res_gp, ax=ax)
    fig.savefig("plots/convergence_{}.pdf".format(cfg["channel"]))

    # print(mva.best_params_)
    # print(mva.best_estimator_)
    mva = XGBClassifier(learning_rate=res_gp.x[0],
                        n_estimators=res_gp.x[1],
                        max_depth=res_gp.x[2],
                        subsample=res_gp.x[3],
                        reg_alpha=res_gp.x[4],
                        reg_lambda=res_gp.x[5],
                        min_child_weight=res_gp.x[6],
                        gamma=res_gp.x[7],
                        n_jobs=18)
    print(mva)
    mva.fit(df_train[features], df_train.Signal,
            sample_weight=df_train.MVAWeight)

    # print()
    # print("Best paramaters:")
    # print(mva.best_params_)
    # print()

    df_test = df_test.assign(MVA=classifiers.evaluate_mva(df_test[features],
                                                          mva))
    df_train = df_train.assign(MVA=classifiers.evaluate_mva(df_train[features],
                                                            mva))
    df = df.assign(MVA=pd.concat((df_train.MVA, df_test.MVA)))

    # Save trained classifier
    # classifiers.save_classifier(mva, cfg, "{}{}_{}".format(cfg["mva_dir"],
    #                                                        cfg["classifier"],
    #                                                        cfg["channel"]))

    # Metrics
    metrics.print_metrics(mva, df_train[features], df_test[features],
                          df_train.Signal, df_test.Signal,
                          df_train.MVA, df_test.MVA,
                          df_train.EvtWeight, df_test.EvtWeight)

    pt.make_response_plot(df_train[df_train.Signal == 1].MVA,
                          df_test[df_test.Signal == 1].MVA,
                          df_train[df_train.Signal == 0].MVA,
                          df_test[df_test.Signal == 0].MVA,
                          df_train[df_train.Signal == 1].EvtWeight,
                          df_test[df_test.Signal == 1].EvtWeight,
                          df_train[df_train.Signal == 0].EvtWeight,
                          df_test[df_test.Signal == 0].EvtWeight,
                          filename="{}response_{}.pdf".format(cfg["plot_dir"],
                                                              cfg["channel"]))
    pt.make_roc_curve(df_train.MVA, df_test.MVA,
                      df_train.Signal, df_test.Signal,
                      df_train.EvtWeight, df_test.EvtWeight,
                      filename="{}roc_{}.pdf".format(cfg["plot_dir"],
                                                     cfg["channel"]))

    # Binning
    def response(x): return classifiers.evaluate_mva(x[features], mva)
    outrange = (0, 1)

    if cfg["root_out"]["strategy"] == "equal":
        bins = cfg["root_out"]["bins"]
    elif cfg["root_out"]["strategy"] == "quantile":
        bins = df.MVA.quantile(np.linspace(0, 1, cfg["root_out"]["bins"] + 1))
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    elif cfg["root_out"]["strategy"] == "recursive_median":
        bins = binning.recursive_median(
            df.MVA, df.Signal, df.EvtWeight,
            s_num_thresh=cfg["root_out"]["min_signal_events"],
            b_num_thresh=cfg["root_out"]["min_background_events"],
            s_err_thresh=cfg["root_out"]["max_signal_error"],
            b_err_thresh=cfg["root_out"]["max_background_error"])
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    elif cfg["root_out"]["strategy"] == "recursive_kmeans":
        _, bins = binning.recursive_kmeans(
            df.MVA.values.reshape(-1, 1), df.Signal, xw=df.EvtWeight,
            s_num_thresh=cfg["root_out"]["min_signal_events"],
            b_num_thresh=cfg["root_out"]["min_background_events"],
            s_err_thresh=cfg["root_out"]["max_signal_error"],
            b_err_thresh=cfg["root_out"]["max_background_error"],
            bin_edges=True, n_jobs=-1)
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    else:
        raise ValueError("Unrecognised value for option 'strategy': ",
                         cfg["root_out"]["strategy"])

    rootIO.write_root(
        cfg["input_dir"], cfg["features"], response,
        selection=cfg["selection"], bins=bins,
        data=cfg["root_out"]["data"], combine=cfg["root_out"]["combine"],
        data_process=cfg["data_process"], drop_nan=cfg["root_out"]["drop_nan"],
        channel=cfg["channel"], range=outrange,
        filename="{}mva_{}.root".format(cfg["root_dir"], cfg["channel"]))


if __name__ == "__main__":
    main()
