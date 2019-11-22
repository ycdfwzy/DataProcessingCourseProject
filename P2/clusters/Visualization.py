import numpy as np
from sklearn.manifold import TSNE
from sklearn import decomposition
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd

def scatter(x, labels):
    # set info
    colors = np.array(labels)
    n_clusters = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", n_clusters))

    # draw pic
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add text
    txts = []
    for i in range(5):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)

    # show pic
    plt.show()
    return

def TSNE_scatter(X, labels):
    x = TSNE().fit_transform(X)
    scatter(x, labels)

def PCA_scatter(X, labels):
    x = decomposition.PCA(n_components=2).fit_transform(X)
    scatter(x, labels)

def Factor_scatter(X, labels):
    x = decomposition.FactorAnalysis(n_components=2).fit_transform(X)
    scatter(x, labels)
