#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, \
    unicode_literals
from config import read_config, cfg
import rootIO
import classifiers
import metrics
import preprocessing
import plotting as pt
from sklearn.model_selection import train_test_split


def main():
    # Read configuration
    read_config()

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"], cfg["mva_dir"])

    # Read samples
    df = rootIO.read_trees()

    features = cfg["features"]

    # Configure preprocessing
    pre = []
    try:
        for p in cfg["preprocessors"]:
            if p["preprocessor"] == "robust_scaler":
                preprocessing.add_robust_scaler(pre, **p["config"])
            if p["preprocessor"] == "standard_scaler":
                preprocessing.add_standard_scaler(pre, **p["config"])
    except KeyError:
        pass

    # Make plots
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    pt.make_variable_histograms(sig_df, bkg_df,
                                "{}vars_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))
    pt.make_corelation_plot(sig_df[features],
                            "{}corr_sig_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))
    pt.make_corelation_plot(bkg_df[features],
                            "{}corr_bkg_{}.pdf".format(cfg["plot_dir"],
                                                       cfg["channel"]))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=cfg["test_fraction"],
                                         random_state=52)

    # Classify
    if cfg["classifier"] == "mlp":
        mva = classifiers.mlp(df_train, pre, features)
    elif cfg["classifier"] == "bdt_ada":
        mva = classifiers.bdt_ada(df_train, pre, features)
    elif cfg["classifier"] == "bdt_xgb":
        mva = classifiers.bdt_xgb(df_train, pre, features)
    elif cfg["classifier"] == "bdt_grad":
        mva = classifiers.bdt_grad(df_train, pre, features)
    elif cfg["classifier"] == "random_forest":
        mva = classifiers.random_forest(df_train, pre, features)

    df_test = df_test.assign(MVA=classifiers.evaluate_mva(df_test[features],
                                                          mva))
    df_train = df_train.assign(MVA=classifiers.evaluate_mva(df_train[features],
                                                            mva))

    # Save trained classifier
    classifiers.save_classifier(mva, "{}{}_{}".format(cfg["mva_dir"],
                                                      cfg["classifier"],
                                                      cfg["channel"]))

    # Metrics
    metrics.print_metrics(df_train, df_test, features, mva)

    pt.make_response_plot(df_train[df_train.Signal == 1],
                          df_test[df_test.Signal == 1],
                          df_train[df_train.Signal == 0],
                          df_test[df_test.Signal == 0],
                          "{}response_{}.pdf".format(cfg["plot_dir"],
                                                     cfg["channel"]))
    pt.make_roc_curve(df_train, df_test,
                      "{}roc_{}.pdf".format(cfg["plot_dir"],
                                            cfg["channel"]))

    rootIO.write_root(lambda df: classifiers.evaluate_mva(df, mva),
                      filename="{}mva_{}.root".format(cfg["root_dir"],
                                                      cfg["channel"]))


if __name__ == "__main__":
    main()
