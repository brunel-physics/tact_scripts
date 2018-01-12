#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, \
    unicode_literals
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

    # Evaluate classifiers
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
        km = pca.fit(df[["MVA1", "MVA2"]])

        response = lambda x: pca.transform(
            np.column_stack((
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))))

        # Calculate histogram range by examining extreme values
        extremes = pca.transform([[0, 0], [0, 1], [1, 0], [1, 1]])
        range = (min(extremes), max(extremes))
    elif cfg["combination"] == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters = 20

        km = KMeans(n_clusters=n_clusters, n_jobs=-1)
        km = km.fit(df[["MVA1", "MVA2"]])
        df = df.assign(kmean=km.predict(df[["MVA1", "MVA2"]]))

        # Create a lookup table mapping the cluster lables to their ranking in
        # S/N ratio
        def s_to_n(x):
            signal = x.loc[x.Process == "tZq"].EvtWeight.sum()
            noise = x.loc[x.Process != "tZq"].EvtWeight.sum()

            return signal / noise

        clusters = [el[1] for el in df.groupby("kmean")]
        lut = {}
        for i, cluster in enumerate(sorted(clusters, key=s_to_n), 0):
            lut[cluster.kmean.iloc[0]] = i

        response = lambda x: np.vectorize(lut.__getitem__)(km.predict(
            np.column_stack((
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)))))

        # Set output range and override the number of bins - cluster label is
        # not continuous.
        range = (0, n_clusters)
        cfg["root_out"]["bins"] = n_clusters

        plotting.make_kmeans_cluster_plots(
            df, km,
            filename1="{}kmeans_areas_{}.pdf"
            .format(cfg["plot_dir"], cfg["channel"]),
            filename2="{}kmeans_clusters_{}.pdf"
            .format(cfg["plot_dir"], cfg["channel"]))
    else:
        raise ValueError("Unrecogised value for option 'combination': ",
                         cfg["combination"])

    rootIO.write_root(response, range=range,
                      filename="{}mva_{}.root".format(cfg["root_dir"],
                                                      cfg["channel"]))


if __name__ == "__main__":
    main()
