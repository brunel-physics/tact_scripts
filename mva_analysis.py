#!/usr/bin/env python

import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from root_pandas import read_root


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

    return pd.concat(sig_dfs), pd.concat(bkg_dfs)


def make_variable_histograms(sig_df, bkg_df, filename="vars.pdf"):
    """Produce histograms comparing the signal and background distribution
    of availible variables and write them to filename"""

    def plot_histograms(df, ax):
        """Plot histograms for every column in df"""
        return df.hist(bins=100, ax=ax, alpha=0.5, weights=df.EvtWeight,
                       normed=True)

    fig_size = (50, 31)

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)

    ax = plot_histograms(sig_df, ax).flatten()[:len(sig_df.columns)]
    plot_histograms(bkg_df, ax)

    # Save figure
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:  # directory already exists
        pass

    fig.savefig(filename)


def main():
    plt.style.use("ggplot")

    # Configuration
    mz = 50
    mw = 50
    channel = 1  # 0 -> mumu, 1 -> ee
    channel_str = {0: "mumu", 1: "ee"}[channel]
    signals = ["tZq"]

    sig_df, bkg_df = read_trees(signals, channel, mz, mw)
    make_variable_histograms(sig_df, bkg_df,
                             "plots/vars_{}.pdf".format(channel_str))


if __name__ == "__main__":
    main()
