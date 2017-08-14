import numpy as np
np.random.seed(52)
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier


def evaluate_mva(df, mva, training_vars):
    try:
        df = df.assign(MVA=mva.predict_proba(df[training_vars])[:, 1])
    except KeyError:  # Keras doesn't like DataFrames
        df = df.assign(MVA=mva.predict_proba(df[training_vars].as_matrix(),
                                             verbose=0)[:, 1])
    return df


def mlp(df_train, df_test, training_vars, **kwargs):
    """Train using a Multi Layer Perceptron"""

    ann = KerasClassifier(**kwargs)
    ann.fit(df_train[training_vars].as_matrix(), df_train.Signal.as_matrix(),
            sample_weight=df_train.MVAWeight.as_matrix(),
            callbacks=[EarlyStopping(monitor="loss",
                                     min_delta=0,
                                     patience=1,
                                     verbose=0,
                                     mode="auto")])

    return ann


def bdt_ada(df_train, df_test, training_vars, **kwargs):
    """Train using an AdaBoosted Decision Tree"""

    bdt = AdaBoostClassifier(**kwargs)
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight.as_matrix())

    return bdt


def bdt_grad(df_train, df_test, training_vars, **kwargs):
    """Train using a Gradient Boosted Decision Tree"""

    bdt = GradientBoostingClassifier(**kwargs)
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight)

    return bdt


def bdt_xgb(df_train, df_test, training_vars, **kwargs):
    """Train using an XGBoost Boosted Decision Tree"""

    bdt = XGBClassifier(**kwargs)

    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight,)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test[training_vars], df_test.Signal)])

    return bdt


def random_forest(df_train, df_test, training_vars, **kwargs):
    """Train using a Random Forest"""

    rf = RandomForestClassifier(**kwargs)
    rf.fit(df_train[training_vars], df_train.Signal,
           sample_weight=df_train.MVAWeight.as_matrix())

    return rf
