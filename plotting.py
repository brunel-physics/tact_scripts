# -*- coding: utf-8 -*-

"""
This module contains functions which create plots.

These deal only with plots created with matplotlib. Functions handling the
creation of ROOT histograms are found in the rootIO module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from operator import sub

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import auc, roc_curve

from config import cfg


def make_variable_histograms(sig_df, bkg_df, filename="vars.pdf"):
    """
    Produce histograms comparing the distribution of features in sig_df and
    bkg_df.

    Histograms are only produced for columns listed in the features config
    option.

    Parameters
    ----------
    sig_df : DataFrame
        DataFrame containing signal data.
    bkg_df : DataFrame
        DataFrame containing background data.
    filename : string, optional
        Name of the file the plot is saved to.

    Returns
    -------
    None
    """

    def plot_histograms(df, ax):
        """Plot histograms for every column in df"""
        return df[features].hist(bins=42, ax=ax, alpha=0.5,
                                 weights=df.EvtWeight, normed=True)

    features = cfg["features"]

    plt.style.use("ggplot")

    ncols = 2
    nrows = len(features) // ncols + 1

    fig_size = (ncols * 1.618 * 3, nrows * 3)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    fig.set_size_inches(fig_size)

    ax = ax.flatten()

    for i in xrange(1, len(features) % ncols + 1):
        ax[-i].remove()

    ax = ax[:len(features)]

    ax = plot_histograms(sig_df, ax)
    plot_histograms(bkg_df, ax)

    for axis in ax:
        axis.legend(["Signal", "Background"], fontsize="x-small")

    fig.savefig(filename)


def make_corelation_plot(df, filename="corr.pdf"):
    """
    Produce matshow plot representing the correlation matrix of DataFrame
    df.

    Correlation coefficients are calculated between every column in df.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing data for which the correlation coefficients are to
        be calculated.
    filename : string, optional
        Name of the file the plot is saved to.

    Returns
    -------
    None
    """

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

    fig.savefig(filename)


def make_response_plot(sig_df_train, sig_df_test, bkg_df_train, bkg_df_test,
                       bins=25, filename="response.pdf"):
    """
    Produce histogram comparing the response of the test data and training data
    in signal and background.

    Every provided DataFrame must contain a column named "MVA" containing the
    response.

    Parameters
    ----------
    sig_df_train : DataFrame
        Dataframe containing signal training data.
    sig_df_test : DataFrame
        Dataframe containing signal testing data.
    bkg_df_train : DataFrame
        Dataframe containing background training data.
    bkg_df_test : DataFrame
        Dataframe containing background testing data.
    bins : int, optional
        Number of bins in histogram
    filename : string, optional
        Name of the file the plot is saved to.
    """

    plt.style.use("ggplot")

    x_range = (0, 1)

    fig, ax = plt.subplots()

    # Plot histograms of test samples
    for df, label in ((sig_df_test, "Signal (test sample)"),
                      (bkg_df_test, "Background (test sample)")):
        ax = df.MVA.plot.hist(bins=bins, ax=ax, weights=df.EvtWeight,
                              normed=True, range=x_range, alpha=0.5,
                              label=label)

    plt.gca().set_prop_cycle(None)  # use the same colours again

    # Plot error bar plots of training samples
    for df, label in ((sig_df_train, "Signal (training sample)"),
                      (bkg_df_train, "Background (training sample)")):
        hist, bin_edges = np.histogram(df.MVA, bins=bins, range=x_range,
                                       weights=df.EvtWeight)
        hist2 = np.histogram(df.MVA, bins=bins, range=x_range,
                             weights=df.EvtWeight.pow(2))[0]
        db = np.array(np.diff(bin_edges), float)
        yerr = np.sqrt(hist2) / db / hist.sum()
        hist = hist / db / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.errorbar(bin_centers, hist, fmt=",", label=label,
                    yerr=yerr, xerr=(-sub(*x_range) / bins / 2))

    ax.legend(fontsize="small")

    fig.savefig(filename)


def make_roc_curve(df_train, df_test, filename="roc.pdf"):
    """
    Plot the receiver operating characteristic curve for the test and training
    data.

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    df_test : DataFrame
        DataFrame containing testing data.
    filename : string, optional
        Name of the file the plot is saved to.

    Returns
    -------
    None
    """

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, df in (("train", df_train), ("test", df_test)):
        fpr[i], tpr[i], _ = roc_curve(df.Signal, df.MVA,
                                      sample_weight=df.EvtWeight)
        roc_auc[i] = auc(fpr[i], tpr[i], reorder=True)

    plt.style.use("ggplot")

    fig, ax = plt.subplots()

    for i in fpr:
        ax.plot(fpr[i], tpr[i],
                label="ROC curve for {} set (auc = {:0.2f})"
                .format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.savefig(filename)


def make_scatter_plot(df, col_x="MVA1", col_y="MVA2", col_w="EvtWeight",
                      filename="scatter.pdf"):
    """
    Plot the responses of two classifiers for every item in df on a scatter
    plot.

    Events will be coloured red, if they are signal in the first
    classifier, blue if they are background in the first classifier, and green
    otherwise.

    The size of points will be scaled according to the absolute value of the
    associated weight.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing data to be displayed.
    col_x : string, optional
        Name of the column in df whose values should be used as the x-position
        on the 2D plane.
    col_y : string, optional
        Name of the column in df whose values should be used as the y-position
        on the 2D plane.
    col_w : string, optional
        Name of the column in df whose values should be used as the sample
        weights.
    filename : string, optional
        Name of the file the plot is saved to.

    Returns
    -------
    None
    """

    plt.style.use("ggplot")

    fig, ax = plt.subplots()

    df.plot.scatter(
        col_x, col_y, ax=ax, marker=',', s=df[col_w].abs(),
        c=np.select([np.in1d(df.Process, cfg["mva1"]["signals"]),
                     np.in1d(df.Process, cfg["mva1"]["whitelist"])],
                    ["#e24a33", "#8eba42"],
                    default="#348abd"))

    fig.savefig(filename)


def make_kmeans_cluster_plots(df, km, col_x="MVA1", col_y="MVA2",
                              col_label="kmean", col_w="EvtWeight",
                              filename1="kmeans_areas.pdf",
                              filename2="kmeans_clusters.pdf"):
    """
    Plot the result of kmeans clustering. This produces a scatter plot similar
    to make_scatter_plot but colour-coded by cluster and a plot showing
    the extent of each cluster on the 2D plane.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing data to be displayed.
    km
        Trained kmeans classifier
    col_x : string, optional
        Name of column in df containing observations for the x-axis.
    col_y : string, optional
        Name of column in df containing observations for the y-axis.
    col_label : string, optional
        Name of column in df containing cluster labels.
    col_w : string, optional
        Name of column in df containing sample weights.
    filename1, filename2 : string, optional
        Name of the files the plots are saved to.

    Returns
    -------
    None
    """

    # First plot shows the full extent of each cluster
    fig1, ax1 = plt.subplots()

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    h = (x_max - x_min) / 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax1.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap="tab20",
               aspect='auto', origin='lower')

    ax1.grid(False)

    fig1.savefig(filename1)

    # Second plot shows the usual 2D scatter plot, but colour is based on
    # cluster membership
    fig2, ax2 = plt.subplots()

    df.plot.scatter(col_x, col_y, marker=",", c=col_label, ax=ax2,
                    s=df[col_w].abs(), cmap="tab20", colorbar=False)

    fig2.savefig(filename2)
