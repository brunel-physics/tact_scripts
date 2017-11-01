#!/usr/bin/env python

from __future__ import print_function
import sys
from config import read_config, cfg
import rootIO
import preprocessing
import classifiers
import metrics
import plotting as pt
from sklearn.model_selection import train_test_split


def main():
    # Read configuration
    read_config(sys.argv[1])
    training_vars = cfg["training_vars"]

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"])

    # Read samples
    df = rootIO.read_trees()

    # Preprocess data
    df[training_vars], sc = preprocessing.robust_scale(df[training_vars])

    # Make plots
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    pt.make_variable_histograms(sig_df, bkg_df,
                                "{}vars_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))
    pt.make_corelation_plot(sig_df[training_vars],
                            "{}corr_sig_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))
    pt.make_corelation_plot(bkg_df[training_vars],
                            "{}corr_bkg_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=cfg["test_fraction"],
                                         random_state=52)

    # Classify
    if cfg["classifier"] == "mlp":
        mva = classifiers.mlp(df_train, df_test, training_vars)
    elif cfg["classifier"] == "bdt_ada":
        mva = classifiers.bdt_ada(df_train, df_test, training_vars)
    elif cfg["classifier"] == "bdt_xgb":
        mva = classifiers.bdt_xgb(df_train, df_test, training_vars)
    elif cfg["classifier"] == "bdt_grad":
        mva = classifiers.bdt_grad(df_train, df_test, training_vars)
    elif cfg["classifier"] == "random_forest":
        mva = classifiers.random_forest(df_train, df_test, training_vars)

    df_test = classifiers.evaluate_mva(df_test, mva, training_vars)
    df_train = classifiers.evaluate_mva(df_train, mva, training_vars)

    # Metrics
    metrics.print_metrics(df_train, df_test, training_vars, mva)

    pt.make_response_plot(df_train[df_train.Signal == 1],
                          df_test[df_test.Signal == 1],
                          df_train[df_train.Signal == 0],
                          df_test[df_test.Signal == 0],
                          mva,
                          "{}response_{}.pdf".format(cfg["plot_dir"],
                                                     cfg["channel"]))
    pt.make_roc_curve(df_train, df_test,
                      "{}roc_{}.pdf".format(cfg["plot_dir"],
                                            cfg["channel"]))

    rootIO.write_root(mva, scaler=sc,
                      filename="{}mva_{}.root".format(cfg["root_dir"],
                                                      cfg["channel"]))


if __name__ == "__main__":
    main()
