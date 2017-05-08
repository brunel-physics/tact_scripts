# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import glob
import ROOT
import numpy as np
import pandas as pd
from root_numpy import fill_hist
from root_pandas import read_root
from classifiers import evaluate_mva
from more_itertools import unique_everseen


def read_tree(root_file, tree, channel):
    """
    Read a Ttree into a DataFrame

    Parameters
    ----------
    root_file : string
        Path of root file to be read in
    tree : string
        Name of tree to be read in
    channel : "ee" or "mumu"
        Channel to be read in

    Returns
    -------
    df : DataFrame
        DataFrame containing data read in from Ttree
    """

    # Read ROOT trees into data frames
    try:
        df = read_root(root_file, tree)
    except IOError:  # occasional failure for empty trees
        return pd.DataFrame()

    return df[df.Channel == {"ee": 1, "mumu": 0}[channel]]  # filter channel


def read_trees(signals, channel, mz, mw, blacklist=(),
               negative_weight_treatment="reweight"):
    """Read in TTrees with Z mass cut mw and W mass cut mw"""

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

    root_files = glob.iglob("/scratch/data/TopPhysics/mvaDirs/inputs/2016/all/"
                            "mz{}mw{}/*.root".format(mz, mw))

    for root_file in root_files:
        process = get_process_name(root_file)

        # Ignore any samples matching any pattern in blacklist
        if any(re.match(pattern, process) for pattern in blacklist):
            continue

        df = read_tree(root_file, "Ttree_{}".format(process), channel)

        if df.empty:
            continue

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


def _format_TH1_name(name, channel, combine=True):
    """
    Modify name of Ttrees from input files to a format expected by combine
    or THETA

    Parameters
    ----------
    name : string
        Name of the Ttree.
    channel : "ee" or "mumu"
        The channel contained within the histogram
    combine : bool, optional
        Whether the names should be in a combine-compatible format

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

    name = re.sub(r"^Ttree", "MVA_{}_".format(channel), name)
    if combine:
        name = re.sub(r"__plus$", "Up", name)
        name = re.sub(r"__minus$", "Down", name)

    return name


def MVA_to_TH1(df, bins=100, name="MVA", title="MVA"):
    """
    Write MVA discriminant from a DataFrame to a TH1D

    Parameters
    ----------
    df : DataFrame
        Dataframe contaning an "MVA" column containing the MVA discriminant and
        "EvtWeight" column containing event weights.
    bins : int, optional
        Number of bins in the TH1.
    name : string, optional
        Name of TH1.
    title : string, optional
        Title of TH1.

    Returns
    -------
    h : TH1D
        TH1D of MVA discriminant.
    """

    h = ROOT.TH1D(name, title, bins, df.MVA.min(), df.MVA.max())
    fill_hist(h, df.MVA.as_matrix(), df.EvtWeight.as_matrix())
    return h


def poisson_pseudodata(df):
    """
    Generate Poisson pseudodata from a DataFrame by binning the MVA
    discriminant in a TH1D and applying a Poisson randomisation to each bin.

    Paramaters
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


def write_root(mva, channel, mz, mw, training_vars,
               filename="mva.root", data="poisson", combine=True):
    """
    Evaluate an MVA and write the result to TH1s in a root file.

    Parameters
    ----------
    mva : trained classifier
        Classfier on which read-in Ttrees will be evaluated.
    channel : "ee" or "mumu"
        The channel on which the classifier will be evaluated.
    mz : int
        Z mass cut in GeV.
    mw : int
        W mass cut in GeV.
    training_vars: array_like
        Names of features on which the mva was trained.
    filename : string, optional
        Name of the output root file (including directory).
    data : string, optional
        Pseudodata generation method

        "empty"
        Pseudodata histogram is empty.

        "poisson"
        Pseudodata is a TH1 containing all non-systematic MC data with a
        poisson error applied to each bin.

    combine : bool, optional
        Whether the output root file should have TH1 names compatible with
        the Higgs Combined Analysis tool.

    Returns
    -------
    None
    """

    root_files = glob.iglob("/scratch/data/TopPhysics/mvaDirs/inputs/2016/all/"
                            "mz{}mw{}/*.root".format(mz, mw))

    fo = ROOT.TFile(filename, "RECREATE")
    pseudo_dfs = []  # list of dataframes we'll turn into pseudodata
    data_name = "DataEG" if channel == "ee" else "DataMu"

    for root_file in root_files:
        fi = ROOT.TFile(root_file, "READ")

        # Dedupe, the input files contain duplicates for some reason...
        for tree in unique_everseen(key.ReadObj().GetName()
                                    for key in fi.GetListOfKeys()):
            df = read_tree(root_file, tree, channel)

            if df.empty:
                continue

            print("Evaluating classifier on Ttree", tree)
            df = evaluate_mva(df, mva, training_vars)

            # Trees used in pseudodata should be not systematics and not data
            if not re.search(r"(minus)|(plus)|({})$".format(data_name), tree):
                pseudo_dfs.append(df)

            tree = _format_TH1_name(tree, channel, combine)
            h = MVA_to_TH1(df, name=tree, title=tree)
            h.SetDirectory(fo)
            fo.cd()
            h.Write()

    data_process = "data_obs" if combine else "DATA"

    h = ROOT.TH1D()
    if data == "poisson":
        h = poisson_pseudodata(pd.concat(pseudo_dfs))
    elif data == "empty":
        h = ROOT.TH1D()
    else:
        raise ValueError("Unrecogised value for option 'data': ", data)

    h.SetName("MVA_{}__{}".format(channel, data_process))
    h.SetDirectory(fo)
    fo.cd()
    h.Write()

    fo.Close()
