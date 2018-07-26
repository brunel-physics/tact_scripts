# -*- coding: utf-8 -*-

"""
mva_analysis.py

Usage:
    tact config.yaml
or  tact --stdin < config.yaml
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tact import binning, classifiers, config, metrics
from tact import plotting as pt
from tact import preprocessing, rootIO


def main():
    # Read configuration
    try:
        config.read_config()
    except IndexError:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    cfg = config.cfg

    np.random.seed(cfg["seed"])
    plt.style.use("ggplot")

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"], cfg["mva_dir"])

    # Read samples
    df = rootIO.read_trees(
        cfg["input_dir"], cfg["features"], cfg["signals"], cfg["backgrounds"],
        selection=cfg["selection"],
        negative_weight_treatment=cfg["negative_weight_treatment"],
        equalise_signal=cfg["equalise_signal"])

    features = cfg["features"]

    # Make plots
    # sig_df = df[df.Signal == 1]
    # bkg_df = df[df.Signal == 0]
    #
    # pt.make_variable_histograms(df[features], df.Signal, w=df.EvtWeight,
    #                             bins=42, filename="{}vars_{}.pdf"
    #                             .format(cfg["plot_dir"], cfg["channel"]))
    # pt.make_corelation_plot(sig_df[features], w=sig_df.MVAWeight,
    #                         filename="{}corr_sig_{}.pdf"
    #                         .format(cfg["plot_dir"], cfg["channel"]))
    # pt.make_corelation_plot(bkg_df[features], w=bkg_df.MVAWeight,
    #                         filename="{}corr_bkg_{}.pdf"
    #                         .format(cfg["plot_dir"], cfg["channel"]))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=cfg["test_fraction"],
                                         stratify=df.Process)

    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    # Classify
    format_str = "{{:>{}}}".format(max(map(len, features)))
    while len(features) > 1:
        mva = XGBClassifier(n_jobs=20)

        mva.fit(df_train[features], df_train.Signal,
                sample_weight=df_train.MVAWeight)
        feature_importances = mva.feature_importances_

        min_index = np.argmin(feature_importances)

        print(format_str.format(features[min_index]), '\t', "{0:.4e}".format(feature_importances[min_index]), '\t', "{0:.4f}".format(
            roc_auc_score(df_test.Signal, mva.predict_proba(df_test[features])[:, 1], sample_weight=df_test.MVAWeight)))

        del features[min_index]
        # print("Feature importance:")
        # for var, importance in sorted(
        #         zip(list(df_train), feature_importances),
        #         key=lambda x: x[1],
        #         reverse=True):
        #     print("{0:15} {1:.3E}".format(var, importance))
    print(features)


if __name__ == "__main__":
    main()
