#!/usr/bin/env python

from __future__ import print_function
import classifiers
import rootIO
import plotting
import numpy as np
from config import read_config, cfg


def main():
    # Read configuration
    read_config()

    # Load pickled classifiers
    mva1, cfg["mva1"] = classifiers.load_classifier(open(cfg["classifier1"],
                                                         "rb"))
    mva2, cfg["mva2"] = classifiers.load_classifier(open(cfg["classifier2"],
                                                         "rb"))

    cfg["features"] = list(set(cfg["mva1"]["features"] +
                               cfg["mva2"]["features"]))

    # Read TTrees and evaluate classifiers
    df = rootIO.read_trees()
    df = df.assign(MVA1=classifiers.evaluate_mva(df[cfg["mva1"]["features"]],
                                                 mva1))
    df = df.assign(MVA2=classifiers.evaluate_mva(df[cfg["mva2"]["features"]],
                                                 mva2))

    # Plot classifier responses on 2D plane
    plotting.make_scatter_plot(df, filename="{}scatter_{}.pdf"
                               .format(cfg["plot_dir"], cfg["channel"]))

    # Combine classifier scores
    if cfg["combination"] == "min":
        response = lambda x: np.minimum(
            classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
            classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))
        range = (0, 1)
    elif cfg["combination"] == "max":
        response = lambda x: np.maximum(
            classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
            classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))
        range = (0, 1)
    elif cfg["combination"] == "add":
        response = lambda x: \
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1) + \
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)
        range = (0, 2)
    elif cfg["combination"] == "quadrature":
        response = lambda x: np.sqrt(
            np.square(
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1)) +
            np.square(
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)))
        range = (0, np.sqrt(2))
    elif cfg["combination"] == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1, svd_solver="full")
        mva = np.column_stack(
            (classifiers.evaluate_mva(df[cfg["mva1"]["features"]], mva1),
             classifiers.evaluate_mva(df[cfg["mva2"]["features"]], mva2)))
        pca = pca.fit(mva)

        response = lambda x: pca.transform(
            np.column_stack((
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))))

        # Calculate histogram range by examining extreme values
        extremes = pca.transform([[0, 0], [0, 1], [1, 0], [1, 1]])
        range = (min(extremes), max(extremes))
    else:
        raise ValueError("Unrecogised value for option 'combination': ",
                         cfg["combination"])

    rootIO.write_root(response, range=range,
                      filename="{}mva_{}.root".format(cfg["root_dir"],
                                                      cfg["channel"]))


if __name__ == "__main__":
    main()
