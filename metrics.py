# -*- coding: utf-8 -*-

"""
This module contains functions and helper functions that print classifier
metrics to stdout.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.stats import kstwobign
from sklearn.metrics import classification_report, confusion_matrix

from config import cfg


def print_metrics(df_train, df_test, mva):
    """
    Print metrics for a trained classifier to stdout.

    This will print the classification report from scikit-learn for the test
    and training sample and the confusion matrix for the test and training
    sample. The p-value for the two-sample Kolmogorov-Smirnov test performed on
    the test and training samples will b given for the signal and bacground.
    Finally, if supported by the classifier, feature importances will be shown.

    Parameters
    ----------
    df_test : DataFrame
        DataFrame containing testing data.
    df_train: DataFrame
        DataFrame containing training data.
    mva
        Classifier trained on df_train

    Returns
    -------
    None
    """

    features = cfg["features"]

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

    if hasattr(mva, "feature_importances_"):
        print("Feature importance:")
        for var, importance in sorted(
                zip(features, mva.feature_importances_),
                key=lambda x: x[1],
                reverse=True):
            print("{0:15} {1:.3E}".format(var, importance))
    else:
        pass
    print()


def ecdf(x, xw=None):
    r"""
    Return the empirical cumulative distrbution function (ECDF) for a set of
    observations.

    Parameters
    ----------
    x : array_like
        Observations
    weights : array_like, optional
        The weight of each observation, should be the same length as x. If
        omitted or None, each observation will be given equal weight.

    Returns
    -------
    ecdf : callable
        ECDF

    Notes
    -----
    The ECDF is

    .. math:: F_{n}(x) = \frac{\sum_{i=1}^{n}W_{i}I_{[-\infty,x]}(X_{i})}
                              {\sum_{i=1}^{n}W_{i}}

    where :math:`n` is the number of independent and identically distributed
    observations :math:`X_{i}`, :math:`W_{i}` is the corresponding weight on
    each observation, and

    .. math:: I_{[-\infty,x]}(X_{i}) = \begin{cases}
                                       1 & \text{for }X_{i}\leq x\\
                                       0 & \text{otherwise}
                                       \end{cases}.
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
    Computes the Kolmogorov-Smirnov (KS) statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    a, b : Sequence of 1D ndarrays
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    aw, bw: Sequence of 1D ndarrays, optional
        The weights of each observation in a, b. Must be the same length as the
        associated array of observations. If omitted or None, every measurement
        will be assigned an equal weight.

    Returns
    -------
    D : float
        KS statistic
    p-value : float
        Two-tailed p-value

    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample KS test, the distribution is
    assumed to be continuous.

    This is the two-sided test, one-sided tests are not implemented. The test
    uses the two-sided asymptotic KS distribution.

    If the KS statistic is small or the p-value is high, then we cannot reject
    the hypothesis that the distributions of the two samples are the same.

    This function accounts for weights using the recommendations found in [1].

    Convergence is improved in the large-sample KS distribution by using the
    form found by [2].

    References
    ----------
    [1] J. Monahan, "Numerical Methods of Statistics" 2nd Ed., 2011

    [2] M. A. Stephens "Use of the Kolmogorov-Smirnov, Cramer-Von Mises and
    Related Statistics Without Extensive Tables", Journal of the Royal
    Statistical Society, Series B (Methodological), Vol. 32, No. 1., pp.
    115-122, 1970
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
