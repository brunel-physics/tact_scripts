from xgboost import XGBClassifier

def bdt_xgb(df_train, df_test, training_vars):
    """Train using an XGBoost Boosted Decision Tree"""

    bdt = XGBClassifier()
    bdt.fit(df_train[training_vars], df_train.Signal)

    return bdt
