from xgboost import XGBClassifier

def evaluate_mva(df, mva, training_vars):
    df["MVA"] = mva.predict_proba(df[training_vars])[:, 1]
    return df

def bdt_xgb(df_train, df_test, training_vars):
    """Train using an XGBoost Boosted Decision Tree"""

    bdt = XGBClassifier()
    bdt.fit(df_train[training_vars], df_train.Signal)

    return bdt
