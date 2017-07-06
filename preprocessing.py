from sklearn.preprocessing import StandardScaler, RobustScaler


def standard_scale(df, **kwargs):
    """
    Scale the dataframe df using a scikit-learn StandardScaler

    Paramaters
    ----------
    df : DataFrame
        DataFrame containing each variable to be scaled in a column
    kwargs : keyword arguments
        Keyword arguments to be passed to StandardScaler()

    Returns
    -------
    a : array
        Array containing scaled data
    sc : StandardScaler
        Trained scaler
    """

    sc = StandardScaler()
    a = sc.fit_transform(df)

    return a, sc


def robust_scale(df, **kwargs):
    """
    Scale the dataframe df using a scikit-learn RobustScaler

    Paramaters
    ----------
    df : DataFrame
        DataFrame containing each variable to be scaled in a column
    kwargs : keyword arguments
        Keyword arguments to be passed to RobustScaler()

    Returns
    -------
    a : array
        Array containing scaled data
    sc : RobustScaler
        Trained scaler
    """

    sc = RobustScaler()
    a = sc.fit_transform(df)

    return a, sc
