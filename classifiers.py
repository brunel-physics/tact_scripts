import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate_mva(df, mva, training_vars):
    df = df.assign(MVA=mva.predict_proba(df[training_vars])[:, 1])
    return df


def mlp(df_train, df_test, training_vars):
    """Train using a Multi Layer Perceptron"""

    mlp = MLPClassifier(hidden_layer_sizes=(len(training_vars) // 2))
    mlp.fit(df_train[training_vars], df_train.Signal)

    return mlp


def bdt_ada(df_train, df_test, training_vars):
    """Train using an AdaBoosted Decision Tree"""

    dt = DecisionTreeClassifier()
    bdt = AdaBoostClassifier()
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight.as_matrix())

    return bdt


def bdt_grad(df_train, df_test, training_vars):
    """Train using a Gradient Boosted Decision Tree"""

    bdt = GradientBoostingClassifier(n_estimators=75,
                                     verbose=1,
                                     min_samples_split=0.1,
                                     subsample=2.0/3,
                                     learning_rate=0.05)
    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight)

    return bdt


def bdt_xgb(df_train, df_test, training_vars):
    """Train using an XGBoost Boosted Decision Tree"""

    bdt = XGBClassifier(silent=False)

    bdt.fit(df_train[training_vars], df_train.Signal,
            sample_weight=df_train.MVAWeight,)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test[training_vars], df_test.Signal)])

    return bdt


def random_forest(df_train, df_test, training_vars):
    """Train using a Random Forest"""

    rf = RandomForestClassifier()
    rf.fit(df_train[training_vars], df_train.Signal,
           sample_weight=df_train.MVAWeight.as_matrix())

    return rf
