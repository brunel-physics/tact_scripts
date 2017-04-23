#!/usr/bin/env python

import glob
import re
import os
import classifiers
import pandas as pd
import matplotlib.pyplot as plt
from root_pandas import read_root
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def smart_save_fig(fig, path):
    """Saves fig to path, creating required directories if they do not exist"""
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:  # directory already exists
        pass

    fig.savefig(path)


def read_trees(signals, channel, mz, mw):
    """Read in TTrees with Z mass cut mw and W mass cut mw"""

    def get_process_name(path):
        """Given a path to a TTree, return the name of the process contained"""

        return re.split(r"histofile_|\.", path)[-2]

    root_files = glob.iglob("/scratch/data/TopPhysics/mvaDirs/inputs/2016/all/"
                            "mz{}mw{}/*.root".format(mz, mw))

    sig_dfs = []
    bkg_dfs = []

    for root_file in root_files:
        process = get_process_name(root_file)

        # Read ROOT files into data frames
        try:
            df = read_root(root_file, "Ttree_{}".format(process))
        except IOError:  # empty trees
            continue

        df = df[df.Channel == channel]  # filter channel

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


def make_variable_histograms(sig_df, bkg_df, filename="vars.pdf"):
    """Produce histograms comparing the signal and background distribution
    of availible variables and write them to filename"""

    def plot_histograms(df, ax):
        """Plot histograms for every column in df"""
        return df.hist(bins=100, ax=ax, alpha=0.5, weights=df.EvtWeight,
                       normed=True)

    plt.style.use("ggplot")

    fig_size = (50, 31)

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)

    ax = plot_histograms(sig_df, ax).flatten()[:len(sig_df.columns)]
    plot_histograms(bkg_df, ax)

    smart_save_fig(fig, filename)


def make_corelation_plot(df, filename="corr.pdf"):
    """Produce 2D histogram representing the correlation matrix of dataframe
    df. Written to filename."""

    plt.style.use("ggplot")

    corr = df.corr()
    nvars = len(corr.columns)

    fig, ax = plt.subplots()
    ms = ax.matshow(corr, vmin=-1, vmax=1)

    fig.set_size_inches(1 + nvars / 1.5, 1 + nvars / 1.5)
    plt.xticks(xrange(nvars), corr.columns, rotation=90)
    plt.yticks(xrange(nvars), corr.columns)
    ax.tick_params(axis='both', which='both', length=0)  # hide ticks
    ax.grid(False)

    # Workaround for using colorbars with tight_layout
    # https://matplotlib.org/users/tight_layout_guide.html#colorbar
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(ms, cax=cax)

    plt.tight_layout()

    smart_save_fig(plt, filename)


def print_metrics(df_train, df_test, training_vars, mva):
    """Print some basic metrics for the trained mva"""

    test_prediction = mva.predict(df_test[training_vars])
    train_prediction = mva.predict(df_train[training_vars])

    print "Classification Reports"
    print "Test sample:"
    print classification_report(df_test.Signal, test_prediction,
                                target_names=["background", "signal"])
    print "Training sample:"
    print classification_report(df_train.Signal, train_prediction,
                                target_names=["background", "signal"])

    print "Confusion matrix:"
    print "Test sample:"
    print confusion_matrix(df_test.Signal, test_prediction)
    print "Training sample:"
    print confusion_matrix(df_train.Signal, train_prediction)


def main():
    # Configuration
    mz = 50
    mw = 50
    channel = 1  # 0 -> mumu, 1 -> ee
    channel_str = {0: "mumu", 1: "ee"}[channel]
    signals = ["tZq"]
    plot_dir = "plots/"
    test_fraction = 0.25
    training_vars = ["zMass",
                     "jjdelR",
                     "totVecM",
                     "leadJetEta",
                     "zlb1DelR",
                     "totPt"]

    # Read samples
    df = read_trees(signals, channel, mz, mw)
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    # Make plots
    make_variable_histograms(sig_df, bkg_df,
                             "{}vars_{}.pdf".format(plot_dir, channel_str))
    make_corelation_plot(sig_df,
                         "{}corr_sig_{}.pdf".format(plot_dir, channel_str))
    make_corelation_plot(bkg_df,
                         "{}corr_bkg_{}.pdf".format(plot_dir, channel_str))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=test_fraction,
                                         random_state=42)

    # Classify
    bdt_xgb = classifiers.bdt_xgb(df_train, df_test, training_vars)

    # Metrics
    print_metrics(df_train, df_test, training_vars, bdt_xgb)


if __name__ == "__main__":
    main()
