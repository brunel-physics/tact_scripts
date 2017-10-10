#!/usr/bin/env python

from __future__ import print_function
import rootIO
import preprocessing
import classifiers
import metrics
import plotting as pt
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier


def main():
    # Configuration
    blacklist = (r"^Data", r"aMCatNLO")
    mz = 20
    mw = 20
    region = "signal"
    channel = "ee"
    signals = ["tZq"]
    plot_dir = "plots/"
    root_dir = "root/"
    test_fraction = 0.5
    training_vars = [
        "bTagDisc",
        # "fourthJetEta",
        # "fourthJetPhi",
        # "fourthJetPt",
        # "fourthJetbTag",
        # "jetHt",
        # "jetMass",
        # "jetMass3",
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
        # "nJets",
        # "secJetEta",
        # "secJetPhi",
        # "secJetPt",
        # "secJetbTag",
        # "thirdJetEta",
        # "thirdJetPhi",
        # "thirdJetPt",
        # "thirdJetbTag",
        # "topEta",
        "topMass",
        # "topPhi",
        "topPt",
        # "totEta",
        # "totHt",
        # "totHtOverPt",
        # "totPt",
        # "totPt2Jet",
        # "totPtVec",
        # "totVecM",
        # "w1TopDelPhi",
        # "w1TopDelR",
        # "w2TopDelPhi",
        # "w2TopDelR",
        # "wPairEta",
        "wPairMass",
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
        "wwdelR",
        # "wzdelPhi",
        # "wzdelR",
        # "zEta",
        # "zLepdelPhi",
        # "zLepdelR",
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
        "zlb2DelR",
        # "chi2"
        ]

    # Make ouptut directories
    rootIO.makedirs(plot_dir, root_dir)

    # Read samples
    df = rootIO.read_trees(signals, channel, mz, mw, region,
                           blacklist=blacklist,
                           equalise_signal=True,
                           negative_weight_treatment="passthrough")

    # Preprocess data
    df[training_vars], sc = preprocessing.robust_scale(df[training_vars])

    # Make plots
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    pt.make_variable_histograms(sig_df, bkg_df, training_vars,
                                "{}vars_{}.pdf".format(plot_dir, channel))
    pt.make_corelation_plot(sig_df[training_vars],
                            "{}corr_sig_{}.pdf".format(plot_dir, channel))
    pt.make_corelation_plot(bkg_df[training_vars],
                            "{}corr_bkg_{}.pdf".format(plot_dir, channel))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=test_fraction,
                                         random_state=52)

    # Classify
    def build_model():
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.regularizers import l2

        model = Sequential()
        model.add(Dense(8,
                        activation="sigmoid",
                        input_dim=len(training_vars),
                        activity_regularizer=l2(5e-3)))
                        # bias_regularizer=l2(1e-2),
                        # kernel_regularizer=l2(1e-2)))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="nadam",
                      metrics=["accuracy"])

        return model

    mva = classifiers.mlp(df_train, df_test, training_vars,
                          build_fn=build_model,
                          epochs=10000,
                          verbose=1,
                          batch_size=2048)
    # mva = classifiers.bdt_ada(df_train, df_test, training_vars,
    #                           base_estimator=DecisionTreeClassifier())
    # mva = classifiers.bdt_xgb(df_train, df_test, training_vars, silent=False)
    # mva = classifiers.bdt_grad(df_train, df_test, training_vars,
    #                            n_estimators=100,
    #                            verbose=1,
    #                            min_samples_split=0.1,
    #                            subsample=0.75,
    #                            learning_rate=0.02,
    #                            random_state=52,
    #                            max_depth=5)
    # mva = classifiers.random_forest(df_train, df_test, training_vars)

    df_test = classifiers.evaluate_mva(df_test, mva, training_vars)
    df_train = classifiers.evaluate_mva(df_train, mva, training_vars)

    # Metrics
    metrics.print_metrics(df_train, df_test, training_vars, mva)

    pt.make_response_plot(df_train[df_train.Signal == 1],
                          df_test[df_test.Signal == 1],
                          df_train[df_train.Signal == 0],
                          df_test[df_test.Signal == 0],
                          mva,
                          "{}response_{}.pdf".format(plot_dir, channel))
    pt.make_roc_curve(df_train, df_test,
                      "{}roc_{}.pdf".format(plot_dir, channel))

    rootIO.write_root(mva, channel, mz, mw, region, training_vars, scaler=sc,
                      filename="{}mva_{}.root".format(root_dir, channel),
                      combine=True, drop_nan=True, data="empty")


if __name__ == "__main__":
    main()
