from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.stats import kstwobign
from sklearn.metrics import classification_report, confusion_matrix


def print_metrics(df_train, df_test, features, mva):
    """
    Print metrics for a trained classifier

    Parameters
    ----------
    df_test : DataFrame
        DataFrame containing testing data.
    df_train: DataFrame
        DataFrame containing training data.
    features : array_like
        Names of features on which the classifier was trained.
    mva : trained classifier
        Classifier trained on df_train

    Returns
    -------
    None
    """

    try:
        test_prediction = mva.predict(df_test[features])
        train_prediction = mva.predict(df_train[features])
    except (KeyError, UnboundLocalError):
        test_prediction = mva.predict(df_test[features].as_matrix())
        train_prediction = mva.predict(df_train[features].as_matrix())

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
    print()

    print("KS Test p-value")
    print("Signal:")
    print(ks_2samp(df_train[df_train.Signal == 1].MVA,
                   df_test[df_test.Signal == 1].MVA,
                   df_train[df_train.Signal == 1].EvtWeight,
                   df_test[df_test.Signal == 1].EvtWeight)[1])
    print("Background:")
    print(ks_2samp(df_train[df_train.Signal == 0].MVA,
                   df_test[df_test.Signal == 0].MVA,
                   df_train[df_train.Signal == 0].EvtWeight,
                   df_test[df_test.Signal == 0].EvtWeight)[1])
    print()

    try:
        print("Variable importance:")
        for var, importance in sorted(
                zip(features, mva.feature_importances_),
                key=lambda x: x[1],
                reverse=True):
            print("{0:15} {1:.3E}".format(var, importance))
    except AttributeError:
        pass
    print()


def ecdf(x, xw=None):
    """
    Return the emperical cumulative distrbution function (ECDF) for a set of
    observations.

    Parameters
    ----------
    x : array_like
        Observations
    weights : array_like
        The weight of each observation, should be the same length as x. If
        None, each observation will be given equal weight.

    Returns
    -------
    ecdf : function
        ECDF function
    """

    if xw is None:
        xw = np.ones(len(x))

    # Create a sorted array of measurements where each measurement is
    # assoicated with the sum of the total weights of all the measurements less
    # than or equal to it, with the weights normalised such that they sum to 1
    m = np.vstack(((-np.inf, 0),  # required for values < min(x)
                   np.sort(np.column_stack((x, np.cumsum(xw / np.sum(xw)))),
                           axis=0)))

    # Return a function which gives the value of the ECDF at a given x
    # (vectoried for array-like objects!)
    return lambda v: m[np.searchsorted(m[:, 0], v, side="right") - 1, 1]


def ks_2samp(a, b, aw=None, bw=None):
    """
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    a, b : Sequence of 1D ndarrays
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different
    aw, bw: Sequence of 1D ndarrays
        The weights of each observation in a, b. Must be the same length as the
        associated array of observations. If None, every measurement will be
        assigned an equal weight.

    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value
    """

    # Methodology for weighted Kolmogorov-Smirnov test taken from Numerical
    # Methods of Statistics - J. Monahan

    ab = np.sort(np.concatenate((a, b)))

    D = np.max(np.absolute(ecdf(a, aw)(ab) - ecdf(b, bw)(ab)))

    n1 = len(a) if aw is None else np.sum(aw) ** 2 / np.sum(aw ** 2)
    n2 = len(b) if bw is None else np.sum(bw) ** 2 / np.sum(bw ** 2)

    en = np.sqrt(n1 * n2 / float(n1 + n2))

    p = kstwobign.sf((en + 0.12 + 0.11 / en) * D)  # Stephens (1970)

    return D, p
