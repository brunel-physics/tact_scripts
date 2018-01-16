# -*- coding: utf-8 -*-

"""
This modules contain functions which add scikit-learn preprocessors to lists,
which can later be transformed into a scikit-learn Pipeline.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from sklearn.preprocessing import RobustScaler, StandardScaler


def add_standard_scaler(l, **kwargs):
    """
    Appends a scikit-learn StandardScaler to l.

    Parameters
    ----------
    l : list
        Pipeline to be modified
    kwargs : keyword arguments
        Keyword arguments to be passed to StandardScaler()

    Returns
    -------
    list
        Modified list
    """

    return l.append(StandardScaler(**kwargs))


def add_robust_scaler(l, **kwargs):
    """
    Appends a scikit-learn RobustScaler to l.

    Parameters
    ----------
    l : list
        Pipeline to be modified
    kwargs : keyword arguments
        Keyword arguments to be passed to RobustScaler()

    Returns
    -------
    list
        Modified list
    """

    return l.append(RobustScaler(**kwargs))
