import sys
from collections import namedtuple
import numpy as np
from config import cfg
from sklearn.pipeline import make_pipeline
np.random.seed(52)


def evaluate_mva(df, mva, features):
    # Keras doesn't like DataFrames, error thrown depends on Keras version
    try:
        return mva.predict_proba(df[features])[:, 1]
    except (KeyError, UnboundLocalError):  # Keras doesn't like DataFrames
        return mva.predict_proba(df[features].as_matrix())[:, 1]
    return df


def mlp(df_train, pre, features):
    """Train using a Multi Layer Perceptron"""

    def build_model():
        from keras.models import layer_module

        # Set input layer shape
        cfg["mlp"]["model"]["config"][0]["config"]["batch_input_shape"] \
            = (None, len(features))

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

    mva.fit(df_train[features].as_matrix(), df_train.Signal.as_matrix(),
            kerasclassifier__sample_weight=df_train.MVAWeight.as_matrix(),
            kerasclassifier__callbacks=callbacks)

    return mva


def bdt_ada(df_train, pre, features):
    """Train using an AdaBoosted Decision Tree"""

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    bdt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             **cfg["bdt_ada"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[features], df_train.Signal,
            adaboostclassifier__sample_weight=df_train.MVAWeight.as_matrix())

    return mva


def bdt_grad(df_train, pre, features):
    """Train using a Gradient Boosted Decision Tree"""

    from sklearn.ensemble import GradientBoostingClassifier

    bdt = GradientBoostingClassifier(**cfg["bdt_grad"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[features], df_train.Signal,
            gradientboostingclassifier__sample_weight=df_train.MVAWeight)

    return mva


def bdt_xgb(df_train, pre, features):
    """Train using an XGBoost Boosted Decision Tree"""

    from xgboost import XGBClassifier

    bdt = XGBClassifier(**cfg["bdt_xgb"])

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train[features], df_train.Signal,
            xgboostclassifier__sample_weight=df_train.MVAWeight,)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test[features], df_test.Signal)])

    return mva


def random_forest(df_train, pre, features):
    """Train using a Random Forest"""

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(**cfg["random_forest"])

    mva = make_pipeline(*(pre + [rf]))

    rf.fit(df_train[features], df_train.Signal,
           randomforestclassifier__sample_weight=df_train.MVAWeight)

    return mva


SavedClassifier = namedtuple("SavedClassifier", "cfg mva keras")


def save_classifier(mva, filename="mva"):
    """
    Write a trained classifier pipeline and global cofig to an external file.


    Parameters
    ----------
    mva : trained classifier
        Classifier to be trained
    filename : string, optional
        Name of output file (including directory). Extension will be set
        automatically

    Returns
    -------
    None

    Notes
    -----
    Requires the dill package.
    """

    import dill

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
        File classifier is to be loaded from

    Returns
    -------
    mva: Pipeline
        scikit-learn Pipeline containing full classifier stack
    cfg:
        Configuration associated with mva

    Notes
    -----
    Requires the dill package.
    """

    import dill

    sc = dill.load(f)

    return sc.mva, sc.cfg
