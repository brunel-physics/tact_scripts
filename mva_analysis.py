#!/usr/bin/env python

from __future__ import print_function
import rootIO
import classifiers
import plotting as pt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def print_metrics(df_train, df_test, training_vars, mva):
    """Print some basic metrics for the trained mva"""

    test_prediction = mva.predict(df_test[training_vars])
    train_prediction = mva.predict(df_train[training_vars])

    print("Classification Reports")
    print("Test sample:")
    print(classification_report(df_test.Signal, test_prediction,
                                target_names=["background", "signal"]))
    print("Training sample:")
    print(classification_report(df_train.Signal, train_prediction,
                                target_names=["background", "signal"]))

    print("Confusion matrix:")
    print("Test sample:")
    print(confusion_matrix(df_test.Signal, test_prediction))
    print("Training sample:")
    print(confusion_matrix(df_train.Signal, train_prediction))


def main():
    # Configuration
    blacklist = ("^Data.*",)
    mz = 20
    mw = 50
    channel = 1  # 0 -> mumu, 1 -> ee
    channel_str = {0: "mumu", 1: "ee"}[channel]
    signals = ["tZq"]
    plot_dir = "plots/"
    test_fraction = 0.25
    training_vars = [
        # "bTagDisc",
        # "fourthJetEta",
        # "fourthJetPhi",
        # "fourthJetPt",
        # "fourthJetbTag",
        # "jetHt",
        # "jjdelPhi",
        "jjdelR",
        # "leadJetEta",
        # "leadJetPhi",
        # "leadJetPt",
        # "leadJetbTag",
        # "lep1D0",
        # "lep1Eta",
        # "lep1Phi",
        # "lep1Pt",
        # "lep1RelIso",
        # "lep2D0",
        # "lep2Eta",
        # "lep2Phi",
        # "lep2Pt",
        # "lep2RelIso",
        # "lepEta",
        # "lepHt",
        # "lepMass",
        # "lepPhi",
        # "lepPt",
        # "mTW",
        "met",
        # "nBjets",
        "nJets",
        # "secJetEta",
        # "secJetPhi",
        # "secJetPt",
        # "secJetbTag",
        # "thirdJetEta",
        # "thirdJetPhi",
        # "thirdJetPt",
        # "thirdJetbTag",
        "topEta",
        "topMass",
        # "topPhi",
        # "topPt",
        # "totEta",
        "totHt",
        # "totHtOverPt",
        # "totPt",
        # "totPt2Jet",
        "totPtVec",
        "totVecM",
        # "w1TopDelPhi",
        # "w1TopDelR",
        # "w2TopDelPhi",
        # "w2TopDelR",
        # "wPairEta",
        # "wPairMass",
        # "wPairPhi",
        # "wPairPt",
        # "wQuark1Eta",
        # "wQuark1Phi",
        # "wQuark1Pt",
        # "wQuark2Eta",
        # "wQuark2Phi",
        # "wQuark2Pt",
        # "wQuarkHt",
        # "wTopDelPhi",
        # "wTopDelR",
        # "wwdelPhi",
        # "wwdelR",
        # "wzdelPhi",
        # "wzdelR",
        # "zEta",
        # "zLepdelPhi",
        "zLepdelR",
        "zMass",
        # "zPhi",
        # "zPt",
        # "zQuark1DelPhi",
        # "zQuark1DelR",
        # "zQuark2DelPhi",
        # "zQuark2DelR",
        # "zTopDelPhi",
        # "zTopDelR",
        # "zjminPhi",
        # "zjminR",
        # "zl1Quark1DelPhi",
        # "zl1Quark1DelR",
        # "zl1Quark2DelPhi",
        # "zl1Quark2DelR",
        # "zl1TopDelPhi",
        # "zl1TopDelR",
        # "zl2Quark1DelPhi",
        # "zl2Quark1DelR",
        # "zl2Quark2DelPhi",
        # "zl2Quark2DelR",
        # "zl2TopDelPhi",
        # "zl2TopDelR",
        # "zlb1DelPhi",
        # "zlb1DelR",
        # "zlb2DelPhi",
        # "zlb2DelR"
        ]

    # Read samples
    df = rootIO.read_trees(signals, channel, mz, mw, blacklist, "reweight")
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    # Make plots
    pt.make_variable_histograms(sig_df, bkg_df,
                                "{}vars_{}.pdf".format(plot_dir, channel_str))
    pt.make_corelation_plot(sig_df,
                            "{}corr_sig_{}.pdf".format(plot_dir, channel_str))
    pt.make_corelation_plot(bkg_df,
                            "{}corr_bkg_{}.pdf".format(plot_dir, channel_str))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=test_fraction,
                                         random_state=42)

    # Classify
    # mva = classifiers.bdt_ada(df_train, df_test, training_vars)
    # mva = classifiers.bdt_xgb(df_train, df_test, training_vars)
    mva = classifiers.bdt_grad(df_train, df_test, training_vars)
    # mva = classifiers.mlp(df_train, df_test, training_vars)
    # mva = classifiers.random_forest(df_train, df_test, training_vars)
    for df in [df_train, df_test]:
        df = classifiers.evaluate_mva(df, mva, training_vars)

    # Metrics
    print_metrics(df_train, df_test, training_vars, mva)

    pt.make_response_plot(df_train[df_train.Signal == 1],
                          df_test[df_test.Signal == 1],
                          df_train[df_train.Signal == 0],
                          df_test[df_test.Signal == 0],
                          mva,
                          "{}response_{}.pdf".format(plot_dir, channel_str))


if __name__ == "__main__":
    main()
