from __future__ import division, print_function

import glob
import re
import sys
import warnings
import argparse
from collections import defaultdict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from root_numpy import list_trees
from root_pandas import read_root

plt.style.use('seaborn-whitegrid')
mpl.rcParams.update({"font.family": "serif",
                     "pgf.texsystem": "pdflatex",
                     "pgf.rcfonts": False})

colours = {"Z+jets": "#006699",
           "VV": "#ff9933",
           "VVV": "#993399",
           "Single top": "#ff99cc",
           "tZq": "#999999",
           "ttV": "#339933",
           "tt": "#cc0000",
           "NPL": "#003300"}

groups = {
    "TTHbb": "ttV",
    "TTHnonbb": "ttV",
    "WWW": "VVV",
    "WWZ": "VVV",
    "WZZ": "VVV",
    "ZZZ": "VVV",
    "WW1l1nu2q": "VV",
    "WW2l2nu": "VV",
    "ZZ4l": "VV",
    "ZZ2l2nu": "VV",
    "ZZ2l2q": "VV",
    "WZjets": "VV",
    "WZ3lnu": "VV",
    "WZ2l2q": "VV",
    "WZ1l1nu2q": "VV",
    "ZG2l1g": "VV",
    "TsChan": "Single top",
    "TtChan": "Single top",
    "TbartChan": "Single top",
    "TW": "Single top",
    "TbarW": "Single top",
    "TZQ": "tZq",
    "THQ": "Single top",
    "TTWlnu": "ttV",
    "TTW2q": "ttV",
    "TTZ2l2nu": "ttV",
    "TTZ2q": "ttV",
    "TTG": "ttV",
    "TT" : "tt",
    "TTjets" : "tt",
    "TT2l2v" : "tt",
    "TT1l1v2q" : "tt",
    "TWZ": "Single top",
    "Wjets": "W+jets",
    "DYJetsLLPt0To50": "Z+jets",
    "DYJetsLLPt50To100": "Z+jets",
    "DYJetsLLPt100To250": "Z+jets",
    "DYJetsLLPt250To400": "Z+jets",
    "DYJetsLLPt400To650": "Z+jets",
    "DYJetsLLPt650ToInf": "Z+jets",
    "DYJetsToLLM50": "Z+jets",
    "FakeEG": "NPL",
    "FakeMu": "NPL",
    "DataEG": "Data",
    "DataMu": "Data",
    "MuonEG": "Data",
}

def read_tree(*args, **kwargs):
    # Read ROOT trees into data frames
    try:
        df = read_root(*args, **kwargs)
    except (IOError, IndexError):  # failure for empty trees
        return pd.DataFrame()

    return df


def read_trees(input_dir, selection):
    dfs = []

    root_files = glob.iglob(input_dir + r"*.root")

    for root_file in root_files:
        for tree in list_trees(root_file):
            # Parse tree name
            split_tree = tree.split("__")
            process = split_tree[0].split("Ttree_", 1)[-1]
            if process not in groups.keys():
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
                    root_file, tree, where=selection)

            if df.empty:
                print(process+systematic , "skipped (empty)")
                continue
            print(process + systematic)

            # Label $PROCESS__$SYSTEMATIC
            df = df.assign(Group=groups[process] + systematic)
            df = df.assign(Category=process + systematic)

            dfs.append(df)

    df = pd.concat(dfs)
    df.Category = df.Category.astype('category')
    df.Group = df.Group.astype('category')

    return df.reset_index(drop=True)


def bin_data(df, column, pattern, **kwargs):
    data = df[df.Category.str.match(pattern)][column]
    xmin = max(data.min(), data.quantile(0.25) - 2 * scipy.stats.iqr(data))
    xmax = min(data.max(), data.quantile(0.75) + 2 * scipy.stats.iqr(data))
    return np.histogram(data, bins="doane", range=(xmin, xmax), **kwargs)


def bin_mc(df, column, bins):
    category_bin_counts = {}
    for p in df.Category.unique():
        mask = df.Category == p
        category_bin_counts[p] = np.histogram(
            df[mask][column], weights=df[mask].EvtWeight, bins=bins)[0]

    group_bin_counts = defaultdict(lambda: np.zeros(len(bins) -1))
    for k, v in category_bin_counts.iteritems():
        if len(k.split("__")) == 1 and groups[k] != "Data":
            group_bin_counts[groups[k]] += v

    return category_bin_counts, dict(group_bin_counts)


def total_shape_systematics(shape_systematics, category_bin_counts, processes,
                            era):
    s_bin_counts = defaultdict(lambda: np.zeros((
        2, len(category_bin_counts.itervalues().next()))))

    blacklist = ("THQ__ME",
                 "TWZ__pdf",
                 "TWZ__ME",
                 "TTG__pdf",
                 "TTG__ME",
                 "TTG__isr",
                 "TTG__fsr",
                 "DYJetsToLLM50__isr",
                 "DYJetsToLLM50__fsr",
                 )

    for ss in shape_systematics:
        for p in processes:

            ps = p + "__" + ss

            if ps in blacklist and era == 2017:
                # print("Skipped", ps, "(blacklist)")
                s_bin_counts[ss][0] += category_bin_counts[p]
                s_bin_counts[ss][1] += category_bin_counts[p]
                continue

            try:
                s_bin_counts[ss][0] += category_bin_counts[ps + "Down"]
            except KeyError:
                s_bin_counts[ss][0] += category_bin_counts[p]

            try:
                s_bin_counts[ss][1] += category_bin_counts[ps + "Up"]
            except KeyError:
                s_bin_counts[ss][1] += category_bin_counts[p]

    return s_bin_counts


def total_rate_systematics(rate_systematics, category_bin_counts, processes):
    s_bin_counts = defaultdict(lambda: np.zeros((
        2, len(category_bin_counts.itervalues().next()))))

    for rs, (pattern, rate) in rate_systematics.iteritems():
        for p in processes:
            # If rate not associated with process, set it to 0
            err = rate if re.search(pattern, p) else 0

            s_bin_counts[rs][0] += category_bin_counts[p] * (1 - err)
            s_bin_counts[rs][1] += category_bin_counts[p] * (1 + err)

    return s_bin_counts


def total_systematics(shape_systematics, rate_systematics, category_bin_counts,
                      mc_bin_count, processes, era):

    s_bin_counts = total_shape_systematics(shape_systematics,
                                           category_bin_counts, processes,
                                           era)
    s_bin_counts.update(
        total_rate_systematics(rate_systematics, category_bin_counts,
                               processes))

    for key, val in s_bin_counts.iteritems():
        val = val - mc_bin_count
        # val.sort(axis=0)
        val[0] = val[0].clip(max=0)
        val[1] = val[1].clip(min=0)
        s_bin_counts[key] = val

    # for k, v in s_bin_counts.iteritems():
    #     print(k, np.sum(v, axis=1))

    syst_error = np.sqrt(
        sum(np.square(a) for _, a in s_bin_counts.iteritems()))

    return syst_error


def mask_empty_bins(*bin_counts):
    empty_bin_mask = ~np.any(bin_counts, axis=0)
    return [
        np.ma.array(bin_count, mask=empty_bin_mask)
        for bin_count in bin_counts]


def plot_histogram(groups, group_bin_counts, data_bin_count, syst_error, bins,
                   xlabel, outdir):

    mc_bin_count = sum(v for v in group_bin_counts.values())
    # Mask bins where data and MC is 0
    data_bin_count, mc_bin_count = mask_empty_bins(data_bin_count,
                                                   mc_bin_count)
    data_bin_count_error = np.sqrt(data_bin_count)

    # Start plotting
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, gridspec_kw={"height_ratios": [4, 1]}, sharex=True)

    # Stackplot
    # ax1.set_prop_cycle("color", [plt.cm.tab10(i)
    #                              for i in np.linspace(0, 1, len(groups))])
    bin_centres = (bins[:-1] + bins[1:]) / 2

    # Use the pre-calculated bin counts by having one entry per bin with
    # a weight of the bin count
    ax1.hist(
        [bin_centres] * len(groups),
        stacked=True,
        label=groups,
        weights=[group_bin_counts[g] for g in groups],
        bins=bins,
        color=[colours[g] for g in groups],
        edgecolor='black',
        histtype="stepfilled")[0][-1]

    # Rate plot
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
        prop={'size': 6},
        # bbox_to_anchor=(1.05, 1),
        loc="best",
        # borderaxespad=0.,
        frameon=True,
        fancybox=True,
        facecolor="w",
        framealpha=0.8,
        fontsize="x-large",
    )

    # ax1.set_title("CMS Preliminary", loc="left")
    ax1.tick_params(axis='both', which='both', labelsize="large")
    ax2.tick_params(axis='both', which='both', labelsize="large")
    ax1.set_ylabel("Events", fontsize="x-large")

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
    ax2.set_ylabel("Data/MC", fontsize="x-large")
    ax2.set_xlabel(xlabel, family="monospace", fontsize="x-large")

    # Set axis limits
    ax1.set_ylim(bottom=0)

    plt.tight_layout()

    fig.savefig("{}/{}.pdf".format(outdir, xlabel), pad_inches=0,
                bbox_inches="tight")
    fig.savefig("{}/{}.pgf".format(outdir, xlabel), pad_inches=0,
                bbox_inches="tight")

    ax1.set_ylim(bottom=1)
    ax1.set_yscale("log")

    fig.savefig("{}/{}_log.pdf".format(outdir, xlabel))
    fig.savefig("{}/{}_log.pgf".format(outdir, xlabel))

    plt.close(fig)


def plot_features(args, rate_systematics):
    df = read_trees(args.input, args.selection)

    # Get list of processes (don't get data or systematics) and sort by size
    processes = [
        p for p in df.Category.unique()
        if not re.search(r"^Data|^MuonEG$", p) and len(p.split("__")) == 1]
    processes.sort(key=lambda x: df[df.Category == x].EvtWeight.sum())

    groups = [
        g for g in df.Group.unique()
        if not g == "Data" and len(g.split("__")) == 1]
    groups.sort(key=lambda x: df[df.Group == x].EvtWeight.sum())
    groups.insert(0, groups.pop(groups.index('tZq')))

    # Get list of systematics
    shape_systematics = list({
        re.match(r".*__(.*)(?:Up|Down)", p).group(1)
        for p in df.Category
        if "__" in p})


    for column in list(df):
        print(column)

        # Don't modify event weights
        if column == "EvtWeight":
            continue

        try:
            if not np.issubdtype(df[column].dtype, np.number):
                continue
        except TypeError:  # happens
            continue

        # Bin everything
        data_bin_count, bins = bin_data(df, column, r"^Data|^MuonEG$")
        category_bin_counts, group_bin_counts = bin_mc(df, column, bins)
        mc_bin_count = sum(v for v in group_bin_counts.values())
        syst_error = total_systematics(shape_systematics, rate_systematics,
                                       category_bin_counts, mc_bin_count,
                                       processes, args.era)

        plot_histogram(groups, group_bin_counts, data_bin_count, syst_error,
                       bins, column, args.output)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--selection", "-s", default=None, required=False)
    parser.add_argument("--era", "-e", type=int, choices=(2016, 2017),
                        required=True)
    parser.add_argument("--response", action="store_true")
    args = parser.parse_args()

    rate_systematics = {
        "lumi": (re.compile(r".+"), 0.025 if args.era is 2016 else 0.023),
        "fake_ee": (re.compile(r"^FakeEG$"), 0.3),
        "fake_mumu": (re.compile(r"^FakeMu$"), 0.3),
        "DY_rate": (re.compile(r"^DYJets(?:To)?LL"), 0.1),
        "TT_rate": (re.compile(r"^TT(?:$|[0-9a-z])"), 0.1),
        "TtChan_rate": (re.compile(r"^TtChan$"), 0.1),
        "TbartChan_rate": (re.compile(r"^TbartChan$"), 0.1),
        "TTH_rate": (re.compile(r"^TTH"), 0.1),
        "TTG_rate": (re.compile(r"^TTG"), 0.1),
        "WWW_rate": (re.compile(r"^WWW$"), 0.1),
        "WWZ_rate": (re.compile(r"^WWZ$"), 0.1),
        "WZZ_rate": (re.compile(r"^WZZ$"), 0.1),
        "ZZZ_rate": (re.compile(r"^ZZZ$"), 0.1),
        "WW_rate": (re.compile(r"^WW[0-9a-z]"), 0.1),
        "WZ_rate": (re.compile(r"^WZ[0-9a-z]"), 0.1),
        "ZZ_rate": (re.compile(r"^ZZ[0-9a-z]"), 0.1),
        "ZG_rate": (re.compile(r"^ZG[0-9a-z]"), 0.1),
        "TsChan_rate": (re.compile(r"^TsChan$"), 0.1),
        "TW_rate": (re.compile(r"^TW$"), 0.1),
        "TbarW_rate": (re.compile(r"^TbarW$"), 0.1),
        "THQ_rate": (re.compile(r"^THQ$"), 0.1),
        "TTW_rate": (re.compile(r"^TTW[0-9a-z]"), 0.1),
        "TTZ_rate": (re.compile(r"^TTZ[0-9a-z]"), 0.1),
        "TWZ_rate": (re.compile(r"^TWZ$"), 0.1),
        "Wjets_rate": (re.compile(r"^Wjets$"), 0.1),
    }

    if args.response:
        if parser.selection is not None:
            print("WARN: -s argument is ignored when plotting response",
                  file=syst.stderr)
        pass
    else:
        plot_features(args, rate_systematics)




if __name__ == "__main__":
    main(sys.argv)
