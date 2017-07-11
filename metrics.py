from __future__ import print_function
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, confusion_matrix


def print_metrics(df_train, df_test, training_vars, mva):
    """
    Print metrics for a trained classifier

    Parameters
    ----------
    df_test : DataFrame
        DataFrame containing testing data.
    df_train: DataFrame
        DataFrame containing training data.
    training_vars : array_like
        Names of features on which the classifier was trained.
    mva : trained classifier
        Classifier trained on df_train

    Returns
    -------
    None
    """

    try:
        test_prediction = mva.predict(df_test[training_vars])
        train_prediction = mva.predict(df_train[training_vars])
    except KeyError:
        test_prediction = mva.predict(df_test[training_vars].as_matrix())
        train_prediction = mva.predict(df_train[training_vars].as_matrix())

    print("Classification Reports")
    print("Test sample:")
    print(classification_report(df_test.Signal, test_prediction,
                                target_names=["background", "signal"]))
    print("Training sample:")
    print(classification_report(df_train.Signal, train_prediction,
                                target_names=["background", "signal"]))

    print("Confusion matrix:")
    print("Test sample:")
    print(confusion_matrix(df_test.Signal, test_prediction))
    print("Training sample:")
    print(confusion_matrix(df_train.Signal, train_prediction))
    print()

    print("KS Test p-value")
    print("Signal:")
    print(ks_2samp(df_train[df_train.Signal == 1].MVA,
                   df_test[df_test.Signal == 1].MVA)[1])
    print("Background:")
    print(ks_2samp(df_train[df_train.Signal == 0].MVA,
                   df_test[df_test.Signal == 0].MVA)[1])
    print()

    try:
        print("Variable importance:")
        for var, importance in sorted(
                zip(training_vars, mva.feature_importances_),
                key=lambda x: x[1],
                reverse=True):
            print("{0:15} {1:.3E}".format(var, importance))
    except AttributeError:
        pass
    print()
