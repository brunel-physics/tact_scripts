# -*- coding: utf-8 -*-

"""
This module contains functions which create and train classifiers, as well as
saving them to and reading them from disk.

A classifier function takes a DataFrame containing training data, a list
describing preprocessing steps, and a list of features. It will return a
trained scikit-learn Pipeline containing the preprocessing steps and
classifier.

Classifiers are saved do disk using dill as Python's pickle module does not
correctly serialise Keras classifiers. It should be noted that Keras does not
recommend pickling for neural network serialisation, but no issues have been
observed so far using the dill library.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from collections import namedtuple

import numpy as np
from sklearn.pipeline import make_pipeline

from config import cfg

np.random.seed(52)


def evaluate_mva(df, mva):
    """
    Evaluate the response of a trained classifier.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing features.
    mva
        Trained classifier.

    Returns
    -------
    Series or array
        Classifier response values corresponding to each entry in df.

    Notes
    -----
    The classifier response values are taken from the mva object's
    predict_proba method. By default this is passed the df DataFrame directly
    but in some cases this is not supported and df is passed as a numpy array.
    In the former case this function returns a Pandas Series and in the latter
    a 1D array. This fallback has only been tested for Keras classifiers.
    """

    # Keras doesn't like DataFrames, error thrown depends on Keras version
    try:
        return mva.predict_proba(df)[:, 1]
    except (KeyError, UnboundLocalError):
        return mva.predict_proba(df.as_matrix())[:, 1]


def mlp(df_train, pre):
    """
    Train using a multi-layer perceptron (MLP).

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    pre : list
        List containing preprocessing steps.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.

    Notes
    -----
    This function requires Keras to be availible. Additional configuration can
    be configured using Keras' configuration file. See the Keras documentation
    for more information.

    Keras should outperform scikit-learn's internal MLP implementation in most
    cases, and supports sample weights while training.
    """

    def build_model():
        from keras.models import layer_module

        # Set input layer shape
        cfg["mlp"]["model"]["config"][0]["config"]["batch_input_shape"] \
            = (None, len(cfg["features"]))

        model = layer_module.deserialize(cfg["mlp"]["model"])

        model.compile(**cfg["mlp"]["compile_params"])

        return model

    from keras.wrappers.scikit_learn import KerasClassifier

    callbacks = []
    if cfg["mlp"]["early_stopping"]:
        from keras.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(**cfg["mlp"]["early_stopping_params"]))

    ann = KerasClassifier(build_fn=build_model,
                          **cfg["mlp"]["model_params"])

    mva = make_pipeline(*(pre + [ann]))

    mva.fit(df_train[cfg["features"]].as_matrix(), df_train.Signal.as_matrix(),
            kerasclassifier__sample_weight=df_train.MVAWeight.as_matrix(),
            kerasclassifier__callbacks=callbacks)

    return mva


def bdt_ada(df_train, pre):
    """
    Train using an AdaBoosted decision tree.

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    pre : list
        List containing preprocessing steps.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.
    """

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    bdt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             **cfg["bdt_ada"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[cfg["features"]], df_train.Signal,
            adaboostclassifier__sample_weight=df_train.MVAWeight.as_matrix())

    return mva


def bdt_grad(df_train, pre):
    """
    Train using a gradient boosted tecision tree using scikit-learn's
    internal implementation.

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    pre : list
        List containing preprocessing steps.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.
    """

    from sklearn.ensemble import GradientBoostingClassifier

    bdt = GradientBoostingClassifier(**cfg["bdt_grad"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[cfg["features"]], df_train.Signal,
            gradientboostingclassifier__sample_weight=df_train.MVAWeight)

    return mva


def bdt_xgb(df_train, pre):
    """
    Train using a gradient boosted decision tree with the XGBoost library.

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    pre : list
        List containing preprocessing steps.
    features : list
        List containing the names of features to be trained upon. These should
        correspond to column headings in df_train.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.

    Notes
    -----
    Requires xgboost.
    """

    from xgboost import XGBClassifier

    bdt = XGBClassifier(**cfg["bdt_xgb"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[cfg["features"]], df_train.Signal,
            xgboostclassifier__sample_weight=df_train.MVAWeight,)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test[cfg["features"]], df_test.Signal)])

    return mva


def random_forest(df_train, pre):
    """
    Train using a random forest.

    Parameters
    ----------
    df_train : DataFrame
        DataFrame containing training data.
    pre : list
        List containing preprocessing steps.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.
    """

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(**cfg["random_forest"])

    mva = make_pipeline(*(pre + [rf]))

    rf.fit(df_train[cfg["features"]], df_train.Signal,
           randomforestclassifier__sample_weight=df_train.MVAWeight)

    return mva


def save_classifier(mva, filename="mva"):
    """
    Write a trained classifier pipeline and global config to an external file.

    Parameters
    ----------
    mva : trained classifier
        Classifier to be trained
    filename : string, optional
        Name of output file (including directory). Extension will be set
        automatically.

    Returns
    -------
    None

    Notes
    -----
    Requires dill.
    """

    import dill

    SavedClassifier = namedtuple("SavedClassifier", "cfg mva keras")

    keras = 'kerasclassifier' in mva.named_steps

    # Temporarily boost the recursion limit
    tmp = sys.getrecursionlimit()
    sys.setrecursionlimit(9999)

    dill.dump(SavedClassifier(cfg, mva, keras),
              open("{}.pkl".format(filename), "wb"))

    sys.setrecursionlimit(tmp)


def load_classifier(f):
    """
    Load a trained classifier from a pickle file.

    Parameters
    ----------
    f : file
        File classifier is to be loaded from.

    Returns
    -------
    mva: Pipeline
        Scikit-learn Pipeline containing full classifier stack.
    cfg:
        Configuration associated with mva.

    Notes
    -----
    Requires dill.
    """

    import dill

    sc = dill.load(f)

    return sc.mva, sc.cfg
