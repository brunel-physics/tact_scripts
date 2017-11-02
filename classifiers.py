import numpy as np
from config import cfg
np.random.seed(52)


def evaluate_mva(df, mva, training_vars):
    # Keras doesn't like DataFrames, error thrown depends on Keras version
    try:
        df = df.assign(MVA=mva.predict_proba(df[training_vars])[:, 1])
    except (KeyError, UnboundLocalError):  # Keras doesn't like DataFrames
        df = df.assign(MVA=mva.predict_proba(df[training_vars].as_matrix(),
                                             verbose=0)[:, 1])
    return df


def mlp(df_train, df_test, training_vars):
    """Train using a Multi Layer Perceptron"""

    def build_model():
        from keras.models import layer_module

        # Set input layer shape
        cfg["mlp"]["model"]["config"][0]["config"]["batch_input_shape"] \
            = (None, len(training_vars))

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
    ann.fit(df_train[training_vars].as_matrix(), df_train.Signal.as_matrix(),
            sample_weight=df_train.MVAWeight.as_matrix(),
            callbacks=callbacks)

    return ann


def bdt_ada(df_train, df_test, training_vars):
    """Train using an AdaBoosted Decision Tree"""

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    bdt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                             **cfg["bdt_ada"])
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight.as_matrix())

    return bdt


def bdt_grad(df_train, df_test, training_vars, **kwargs):
    """Train using a Gradient Boosted Decision Tree"""

    from sklearn.ensemble import GradientBoostingClassifier

    bdt = GradientBoostingClassifier(**cfg["bdt_grad"])
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight)

    return bdt


def bdt_xgb(df_train, df_test, training_vars):
    """Train using an XGBoost Boosted Decision Tree"""

    from xgboost import XGBClassifier

    bdt = XGBClassifier(**cfg["bdt_xgb"])

    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight,)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test[training_vars], df_test.Signal)])

    return bdt


def random_forest(df_train, df_test, training_vars):
    """Train using a Random Forest"""

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(**cfg["random_forest"])
    rf.fit(df_train[training_vars], df_train.Signal,
           sample_weight=df_train.MVAWeight.as_matrix())

    return rf
