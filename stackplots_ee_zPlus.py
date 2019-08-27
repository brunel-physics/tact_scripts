from __future__ import division, print_function

import glob
import re
import sys
import warnings
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from root_numpy import list_trees
from root_pandas import read_root

# plt.style.use('seaborn')
mpl.rcParams.update({"pgf.texsystem": "pdflatex"})


def read_tree(*args, **kwargs):
    # Read ROOT trees into data frames
    try:
        df = read_root(*args, **kwargs)
    except (IOError, IndexError):  # failure for empty trees
        return pd.DataFrame()

    return df


def read_trees(input_dir):
    dfs = []

    root_files = glob.iglob(input_dir + r"*.root")

    for root_file in root_files:
        for tree in list_trees(root_file):
            # Parse tree name
            split_tree = tree.split("__")
            process = split_tree[0].split("Ttree_", 1)[-1]

            if process not in {
                    "tZq", "DYToLL_M10To50_aMCatNLO", "DYToLL_M50_aMCatNLO",
                    "FakeEG", "FakeMu", "TbartChan", "TbartW", "THQ", "TsChan",
                    "TT", "TtChan", "ttH", "TTW", "TtW", "TTZ", "TWZ", "Wjets",
                    "WW", "WWW", "WWZ", "WZ", "WZZ", "ZZ", "ZZZ", "DataEG",
                    "DataMu"}:
                continue

            try:
                systematic = "__" + split_tree[1]

                try:
                    pm = split_tree[2]
                    if pm == "plus":
                        systematic = systematic + "Up"
                    elif pm == "minus":
                        systematic = systematic + "Down"
                    else:
                        continue
                except IndexError:
                    pass
            except IndexError:
                systematic = ""

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                df = read_tree(
                    root_file, tree, where="chi2 > 0 && chi2 < 400 && Channel == 0")

            if df.empty:
                # print("Skipped (empty)")
                continue
            print(process + systematic)

            # Label $PROCESS__$SYSTEMATIC
            df = df.assign(Category=process + systematic)

            dfs.append(df)

    df = pd.concat(dfs)
    df.Category = df.Category.astype('category')

    return df.reset_index(drop=True)


def bin_data(df, column, pattern, **kwargs):
    return np.histogram(
        df[df.Category.str.match(pattern)][column], bins="auto", **kwargs)


def bin_mc(df, column, **kwargs):
    category_bin_counts = {}
    for p in df.Category.unique():
        mask = df.Category == p
        category_bin_counts[p] = np.histogram(
            df[mask][column], weights=df[mask].EvtWeight, **kwargs)[0]

    return category_bin_counts


def total_shape_systematics(shape_systematics, category_bin_counts, processes):
    s_bin_counts = defaultdict(lambda: np.zeros((
        2, len(category_bin_counts.itervalues().next()))))

    for ss in shape_systematics:
        for p in processes:

            ps = p + "__" + ss

            # If the histogram associated with the systematic exists, add it
            # to the total, else use the nominal histogram
            try:
                s_bin_counts[ss][0] += category_bin_counts[ps + "Down"]
            except KeyError:
                s_bin_counts[ss][0] += category_bin_counts[p]

            try:
                s_bin_counts[ss][1] += category_bin_counts[ps + "Up"]
            except KeyError:
                s_bin_counts[ss][1] += category_bin_counts[p]

    return s_bin_counts


def total_rate_systematics(rate_systematcs, category_bin_counts, processes):
    s_bin_counts = defaultdict(lambda: np.zeros((
        2, len(category_bin_counts.itervalues().next()))))

    for rs, (pattern, rate) in rate_systematcs.iteritems():
        for p in processes:
            # If rate not associated with process, set it to 0
            err = rate if re.search(pattern, p) else 0

            s_bin_counts[rs][0] += category_bin_counts[p] * (1 - err)
            s_bin_counts[rs][1] += category_bin_counts[p] * (1 + err)

    return s_bin_counts


def total_systematics(shape_systematics, rate_systematics, category_bin_counts,
                      mc_bin_count, processes):

    s_bin_counts = total_shape_systematics(shape_systematics,
                                           category_bin_counts, processes)
    s_bin_counts.update(
        total_rate_systematics(rate_systematics, category_bin_counts,
                               processes))

    for key, val in s_bin_counts.iteritems():
        val = val - mc_bin_count
        val.sort(axis=0)
        val[0] = val[0].clip(max=0)
        val[1] = val[1].clip(min=0)
        s_bin_counts[key] = val

    syst_error = np.sqrt(
        sum(np.square(a) for _, a in s_bin_counts.iteritems()))

    return syst_error


def mask_empty_bins(*bin_counts):
    empty_bin_mask = ~np.any(bin_counts, axis=0)
    return [
        np.ma.array(bin_count, mask=empty_bin_mask)
        for bin_count in bin_counts]


def main(argv):
    indir = ("/scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20_zPlusJets/")
    df = read_trees(indir)

    # Get list of processes (don't get data or systematics) and sort by size
    processes = [
        p for p in df.Category.unique()
        if not re.search(r"^Data", p) and len(p.split("__")) == 1]
    processes.sort(key=lambda x: df[df.Category == x].EvtWeight.sum())

    # Get list of systematics
    shape_systematics = list({
        re.match(r".*__(.*)(?:Up|Down)", p).group(1)
        for p in df.Category
        if "__" in p})

    rate_systematcs = {
        "lumi": (re.compile(r"^(?!Fake).+"), 0.025),
        "fake_ee": (re.compile(r"^FakeEG$"), 0.3),
        "fake_mumu": (re.compile(r"^FakeMu$"), 0.3)}

    for column in list(df):
        print(column)

        # Don't modify event weights
        if column == "EvtWeight":
            continue

        # Clip data to form overflow bins
        try:
            df[column] = df[column].clip(
                lower=df[column].quantile(0.25) -
                1.5 * scipy.stats.iqr(df[column]),
                upper=df[column].quantile(0.75) +
                1.5 * scipy.stats.iqr(df[column]))
        except TypeError:  # not numeric
            continue

        # Bin everything
        data_bin_count, bins = bin_data(df, column, "^Data")
        data_bin_count_error = np.sqrt(data_bin_count)
        category_bin_counts = bin_mc(df, column, bins=bins)
        mc_bin_count = sum(
            v for k, v in category_bin_counts.iteritems() if k in processes)
        syst_error = total_systematics(shape_systematics, rate_systematcs,
                                       category_bin_counts, mc_bin_count,
                                       processes)

        # Mask bins where data and MC is 0
        data_bin_count, mc_bin_count = mask_empty_bins(data_bin_count,
                                                       mc_bin_count)

        # Start plotting
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, gridspec_kw={"height_ratios": [4, 1]}, sharex=True)

        # Stackplot
        ax1.set_prop_cycle("color", [plt.cm.tab20(i)
                                     for i in np.linspace(0, 1, len(processes))])
        ax1.hist(
            [df[df.Category == p][column] for p in processes],
            stacked=True,
            label=processes,
            weights=[df[df.Category == p].EvtWeight for p in processes],
            bins=bins,
            edgecolor='black',
            histtype="stepfilled")[0][-1]

        # Rate plot
        bin_centres = (bins[:-1] + bins[1:]) / 2
        ax1.bar(
            bin_centres,
            np.sum(syst_error, axis=0),
            width=np.diff(bins),
            bottom=mc_bin_count - syst_error[0],
            fill=False,
            color="black",
            linewidth=0,
            label="Syst.",
            hatch="////")

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings(
                "ignore", r"Warning: converting a masked element to nan.")
            ax1.errorbar(
                bin_centres,
                data_bin_count,
                yerr=data_bin_count_error,
                fmt="k.",
                label="Data")
        ax1.legend(
            prop={'size': 5},
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.)

        ax1.set_ylabel("Events")

        ax2.errorbar(
            bin_centres,
            data_bin_count / mc_bin_count,
            yerr=data_bin_count_error / mc_bin_count,
            fmt="k.")
        ax2.bar(
            bin_centres,
            np.sum(syst_error, axis=0) / mc_bin_count,
            width=np.diff(bins),
            bottom=(mc_bin_count - syst_error[0]) / mc_bin_count,
            fill=False,
            color="black",
            linewidth=0,
            label="Syst.",
            hatch="////")

        ax2.set_axisbelow(True)
        ax2.minorticks_on()
        ax2.yaxis.grid(b=True, which='both')
        ax2.set_ylim([0.5, 1.5])
        ax2.set_ylabel("Data/MC ratio")
        ax2.set_xlabel(column)

        plt.tight_layout()

        fig.savefig("plots_ee_zPlus/{}.pdf".format(column))
        #fig.savefig("plots_ee_zPlus/{}.pgf".format(column))

        ax1.set_yscale("log")

        fig.savefig("plots_ee_zPlus/{}_log.pdf".format(column))
        #fig.savefig("plots_ee_zPlus/{}_log.pgf".format(column))

        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv)
