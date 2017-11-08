# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import os
import errno
import glob
from operator import truediv
import ROOT
import numpy as np
import pandas as pd
from root_numpy import array2hist
from root_pandas import read_root
from classifiers import evaluate_mva
from more_itertools import unique_everseen
from config import cfg


def makedirs(*paths):
    """
    Creates a directory for each path given. No effect if the directory
    already exists

    Parameters
    ----------
    paths : strings
        Strings contaning the path of each directory desired to be created

    Returns
    -------
    None
    """

    for path in paths:
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                pass  # directory already exists
            else:
                raise


def read_tree(root_file, tree):
    """
    Read a Ttree into a DataFrame

    Parameters
    ----------
    root_file : string
        Path of root file to be read in
    tree : string
        Name of tree to be read in

    Returns
    -------
    df : DataFrame
        DataFrame containing data read in from Ttree
    """

    Z_MASS = 91.2
    W_MASS = 80.4

    # Read ROOT trees into data frames
    try:
        # We only want to read in the features we're training on, features
        # we're cutting on, and weights
        columns = set(cfg["features"] +
                      ["EvtWeight", "Channel", "wPairMass", "zMass", "chi2"])

        df = read_root(root_file, tree, columns=columns)
    except IOError:  # occasional failure for empty trees
        return pd.DataFrame()

    df = df[(df.Channel == {"ee": 1, "mumu": 0}[cfg["channel"]])  # filter channel
            & (df.zMass.between(Z_MASS - cfg["mz"], Z_MASS + cfg["mz"]))
            & (df.wPairMass.between(W_MASS - cfg["mw"], W_MASS + cfg["mw"]))]

    if cfg["region"] == "all":
        pass
    elif cfg["region"] == "signal":
        df = df[df.chi2 < 40]
    elif cfg["region"] == "control":
        df = df[df.chi2.between(40, 150)]
    else:
        raise ValueError("Unrecogised value for option region: ",
                         cfg["region"])

    return df


def balance_weights(df1, df2):
    """
    Balance the MVA weights in two different DataFrames so they sum to the
    same value.

    Parameters
    ----------
    df1 : DataFrame
        First DataFrame
    df2 : DataFrame
        Second DataFrame

    Returns
    -------
    df1 : DataFrame
         First DataFrame with adjusted weights
    df2 : DataFrame
         Second DataFrame with adjusted weights

    Notes
    -----
    Only one of the returned df1, df2 will have adjusted weights. The function
    will always choose to scale the weights of one DataFrame up to match the
    other.
    """

    sum1 = df1.MVAWeight.sum()
    sum2 = df2.MVAWeight.sum()
    scale = truediv(*sorted([sum1, sum2], reverse=True))  # always scale up

    if sum1 < sum2:
        df1.MVAWeight = df1.MVAWeight * scale
    elif sum1 > sum2:
        df2.MVAWeight = df2.MVAWeight * scale

    assert np.isclose(df1.MVAWeight.sum(), df2.MVAWeight.sum())
    assert df1.MVAWeight.sum() >= sum1
    assert df2.MVAWeight.sum() >= sum2

    return df1, df2


def read_trees():
    """
    Read in Ttrees.

    Parameters
    ----------
    None

    Returns
    -------
    df : DataFrame
        DataFrame containing the Ttree data, MVA weights (as "MVAWeight") and
        classification flag for each event ("Signal" == 1 for signal events,
        0 otherwise)
    """

    def get_process_name(path):
        """Given a path to a TTree, return the name of the process contained"""

        return re.split(r"histofile_|\.", path)[-2]

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
        assert (df.MVAWeight >= 0).all(), "Negative MVA Weights after reweight"

        return df

    sig_dfs = []
    bkg_dfs = []

    root_files = glob.iglob(cfg["input_dir"] + r"*.root")

    for root_file in root_files:
        process = get_process_name(root_file)

        # Ignore any samples matching any pattern in blacklist
        if any(re.search(pattern, process) for pattern in cfg["blacklist"]):
            continue

        df = read_tree(root_file, "Ttree_{}".format(process))

        if df.empty:
            continue

        # Deal with weights
        if cfg["negative_weight_treatment"] == "reweight":
            df = reweight(df)
        elif cfg["negative_weight_treatment"] == "abs":
            df["MVAWeight"] = np.abs(df.EvtWeight)
        elif cfg["negative_weight_treatment"] == "passthrough":
            df["MVAWeight"] = df.EvtWeight
        else:
            raise ValueError("Bad value for option negative_weight_treatment:",
                             cfg["negative_weight_treatment"])

        # Count events
        print("Process ", process, " contains ", len(df.index), " (",
              df.EvtWeight.sum(), ") events", sep='')

        # Split into signal and background
        if process in cfg["signals"]:
            sig_dfs.append(df)
        else:
            bkg_dfs.append(df)

    sig_df = pd.concat(sig_dfs)
    bkg_df = pd.concat(bkg_dfs)

    # Equalise signal and background weights if we were asked to
    if cfg["equalise_signal"]:
        sig_df, bkg_df = balance_weights(sig_df, bkg_df)

    # Label signal and background
    sig_df["Signal"] = 1
    bkg_df["Signal"] = 0

    return pd.concat([sig_df, bkg_df])


def _format_TH1_name(name):
    """
    Modify name of Ttrees from input files to a format expected by combine
    or THETA

    Parameters
    ----------
    name : string
        Name of the Ttree.
    channel : "ee" or "mumu"
        The channel contained within the histogram

    Returns
    -------
    name : The name of the TH1D

    Notes
    -----
    The input name is expected to be in the format:
        Ttree__$PROCESS
    for each process and raw data or
        Ttree__$PROCESS__$SYSTEMATIC__$PLUSMINUS
    for systematics where $PLUSMINUS is plus for 1σ up and minus for 1σ down.
    Ttree is replaced with MVA_$CHANNEL_ and __plus/__minus to Up/Down if the
    combine flag is set.
    """

    name = re.sub(r"^Ttree", "MVA_{}_".format(cfg["channel"]), name)
    if cfg["root_out"]["combine"]:
        name = re.sub(r"__plus$", "Up", name)
        name = re.sub(r"__minus$", "Down", name)

    return name


def MVA_to_TH1(df, name="MVA", title="MVA"):
    """
    Write MVA discriminant from a DataFrame to a TH1D

    Parameters
    ----------
    df : DataFrame
        Dataframe contaning an "MVA" column containing the MVA discriminant and
        "EvtWeight" column containing event weights.
    name : string, optional
        Name of TH1.
    title : string, optional
        Title of TH1.

    Returns
    -------
    h : TH1D
        TH1D of MVA discriminant.
    """

    bins = cfg["root_out"]["bins"]

    contents = np.histogram(df.MVA, bins=bins, range=(0, 1),
                            weights=df.EvtWeight)[0]
    errors, bin_edges = np.histogram(df.MVA, bins=bins, range=(0, 1),
                                     weights=df.EvtWeight.pow(2))
    errors = np.sqrt(errors)

    h = ROOT.TH1D(name, title, len(bin_edges) - 1, bin_edges)
    h.Sumw2()
    array2hist(contents, h, errors=errors)
    return h


def poisson_pseudodata(df):
    """
    Generate Poisson pseudodata from a DataFrame by binning the MVA
    discriminant in a TH1D and applying a Poisson randomisation to each bin.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing the data to be used as a base for the pseudodata.

    Returns
    -------
    h : TH1D
        TH1D contaning pesudodata.
    """

    h = MVA_to_TH1(df)

    for i in xrange(1, h.GetNbinsX() + 1):
        try:
            h.SetBinContent(i, np.random.poisson(h.GetBinContent(i)))
        except ValueError:  # negatve bin
            h.SetBinContent(i, -np.random.poisson(-h.GetBinContent(i)))

    return h


def write_root(mva, filename="mva.root"):
    """
    Evaluate an MVA and write the result to TH1s in a root file.

    Parameters
    ----------
    mva : trained classifier
        Classfier on which read-in Ttrees will be evaluated.
    scaler :
        Scikit-learn scaler used to transform data before being evaluated by
        MVA.
    filename : string, optional
        Name of the output root file (including directory).

    Returns
    -------
    None
    """

    features = cfg["features"]

    root_files = glob.iglob(cfg["input_dir"] + r"*.root")

    fo = ROOT.TFile(filename, "RECREATE")
    pseudo_dfs = []  # list of dataframes we'll turn into pseudodata
    data_name = "DataEG" if cfg["channel"] == "ee" else "DataMu"

    for root_file in root_files:
        fi = ROOT.TFile(root_file, "READ")

        # Dedupe, the input files contain duplicates for some reason...
        for tree in unique_everseen(key.ReadObj().GetName()
                                    for key in fi.GetListOfKeys()):
            df = read_tree(root_file, tree)

            if df.empty:
                continue

            print("Evaluating classifier on Ttree", tree)
            df = evaluate_mva(df, mva, features)

            # Look for and handle NaN Event Weights:
            nan_weights = df.EvtWeight.isnull().sum()
            if nan_weights > 0:
                print("WARNING:", nan_weights, "NaN weights found")
                if cfg["root_out"]["drop_nan"]:
                    df = df[pd.notnull(df["EvtWeight"])]

            # Trees used in pseudodata should be not systematics and not data
            if not re.search(r"(minus)|(plus)|({})$".format(data_name), tree):
                pseudo_dfs.append(df)

            tree = _format_TH1_name(tree)
            h = MVA_to_TH1(df, name=tree, title=tree)
            h.SetDirectory(fo)
            fo.cd()
            h.Write()

    data_process = "data_obs" if cfg["root_out"]["combine"] else "DATA"

    h = ROOT.TH1D()
    h.Sumw2()
    if cfg["root_out"]["data"] == "poisson":
        h = poisson_pseudodata(pd.concat(pseudo_dfs))
    elif cfg["root_out"]["data"] == "empty":
        h = ROOT.TH1D()
    else:
        raise ValueError("Unrecogised value for option 'data': ",
                         cfg["root_out"]["data"])

    h.SetName("MVA_{}__{}".format(cfg["channel"], data_process))
    h.SetDirectory(fo)
    fo.cd()
    h.Write()

    fo.Close()
