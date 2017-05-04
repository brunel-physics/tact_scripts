#!/usr/bin/env python

from __future__ import print_function
import glob
import re
import classifiers
import numpy as np
import pandas as pd
import plotting as pt
from root_pandas import read_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def read_trees(signals, channel, mz, mw, blacklist=(),
               negative_weight_treatment="reweight"):
    """Read in TTrees with Z mass cut mw and W mass cut mw"""

    def get_process_name(path):
        """Given a path to a TTree, return the name of the process contained"""

        return re.split(r"histofile_|\.", path)[-2]

    root_files = glob.iglob("/scratch/data/TopPhysics/mvaDirs/inputs/2016/all/"
                            "mz{}mw{}/*.root".format(mz, mw))

    def reweight(df):
        """
        Takes the abs() of every EvtWeight in a data frame, and scales the
        resulting weights to compensate
        """

        df["MVAWeight"] = np.abs(df.EvtWeight)
        try:
            df["MVAWeight"] = df.MVAWeight * \
                    (df.EvtWeight.sum() / df.MVAWeight.sum())
        except ZeroDivisionError:  # all weights are 0 or df is empty
            pass

        assert np.isclose(df.EvtWeight.sum(), df.MVAWeight.sum()), \
            "Bad weight renormalisation"
        assert (df.MVAWeight >= 0).all(), "Negative MVA Weights"

        return df

    sig_dfs = []
    bkg_dfs = []

    for root_file in root_files:
        process = get_process_name(root_file)

        # Ignore any samples matching any pattern in blacklist
        if any(re.match(pattern, process) for pattern in blacklist):
            continue

        # Read ROOT files into data frames
        try:
            df = read_root(root_file, "Ttree_{}".format(process))
        except IOError:  # occasional failure for empty trees
            continue

        df = df[df.Channel == channel]  # filter channel

        # Deal with weights
        if negative_weight_treatment == "reweight":
            df = reweight(df)
        elif negative_weight_treatment == "abs":
            df["MVAWeight"] = np.abs(df.EvtWeight)
        elif negative_weight_treatment == "passthrough":
            df["MVAWeight"] = df.EvtWeight
        else:
            raise ValueError("Bad value for option negative_weight_treatment:",
                             negative_weight_treatment)

        # Count events
        print("Process ", process, " contains ", len(df.index), " (",
              df.EvtWeight.sum(), ") events", sep='')

        # Split into signal and background
        if process in signals:
            sig_dfs.append(df)
        else:
            bkg_dfs.append(df)

    sig_df = pd.concat(sig_dfs)
    bkg_df = pd.concat(bkg_dfs)

    # Label signal and background
    sig_df["Signal"] = 1
    bkg_df["Signal"] = 0

    return pd.concat([sig_df, bkg_df])


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
    df = read_trees(signals, channel, mz, mw, blacklist, "reweight")
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
