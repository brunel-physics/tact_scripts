from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def evaluate_mva(df, mva, training_vars):
    try:
        df["MVA"] = mva.decision_function(df[training_vars])
    except AttributeError:
        df["MVA"] = mva.predict_proba(df[training_vars])[:, 1]
    return df


def mlp(df_train, df_test, training_vars):
    """Train using a Multi Layer Perceptron"""

    mlp = MLPClassifier(hidden_layer_sizes=(len(training_vars) // 2))
    mlp.fit(df_train[training_vars], df_train.Signal)

    return mlp


def bdt_grad(df_train, df_test, training_vars):
    """Train using a Gradient Boosted Decision Tree"""

    bdt = GradientBoostingClassifier()
    bdt.fit(df_train[training_vars], df_train.Signal)

    return bdt


def bdt_xgb(df_train, df_test, training_vars):
    """Train using an XGBoost Boosted Decision Tree"""

    bdt = XGBClassifier()
    bdt.fit(df_train[training_vars], df_train.Signal)

    return bdt
