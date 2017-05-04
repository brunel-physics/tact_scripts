from __future__ import print_function
import re
import glob
import numpy as np
import pandas as pd
from root_pandas import read_root


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
        assert (df.MVAWeight >= 0).all(), "Negative MVA Weights after reweight"

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
