import numpy as np
np.random.seed(52)
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
                             RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1_l2

def evaluate_mva(df, mva, training_vars):
    try:
        df = df.assign(MVA=mva.predict_proba(df[training_vars])[:, 1])
    except KeyError:  # Keras doesn't like DataFrames
        df = df.assign(MVA=mva.predict_proba(df[training_vars].as_matrix(),
                       verbose=0)[:, 1])
    return df


def mlp(df_train, df_test, training_vars):
    """Train using a Multi Layer Perceptron"""

    def build_model():
        model = Sequential()
        model.add(Dense(10,
            activation="sigmoid",
            input_dim=len(training_vars),
            activity_regularizer=l1_l2(1e-5),
            # kernel_regularizer=l1_l2(1e-4),
            ))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="nadam",
                      metrics=["accuracy"])

        return model

    mlp = KerasClassifier(build_fn=build_model, epochs=10000, verbose=1,
                          batch_size=2048)
    mlp.fit(df_train[training_vars].as_matrix(), df_train.Signal.as_matrix(),
            sample_weight=df_train.MVAWeight.as_matrix(),
            callbacks=[EarlyStopping(monitor="loss",
                                     min_delta=0,
                                     patience=5,
                                     verbose=0,
                                     mode="auto")])

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

    bdt = GradientBoostingClassifier(n_estimators=100,
                                     verbose=1,
                                     min_samples_split=0.1,
                                     subsample=0.75,
                                     learning_rate=0.02,
                                     random_state=52,
                                     max_depth=5)
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
