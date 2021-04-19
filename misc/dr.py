import anndata
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def plot_dr(adata_path, umap_path):
    with open(adata_path, "rb") as f:
        adata = pkl.load(f)
    with open(umap_path, "rb") as f:
        umap_emb = pkl.load(f)

    # implict assertion
    adata.obsm["umap"] = umap_emb
    unique_class = np.unique(adata.obs["bulk_labels"])

    for cls in unique_class:
        plt.scatter(umap_emb[list(adata.obs["bulk_labels"] == cls), 0], umap_emb[list(adata.obs["bulk_labels"] == cls), 1], label=cls, s=0.2)
    plt.legend()
    plt.title("UMAP Embedding")
    plt.savefig("../fig/umap.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str)
    parser.add_argument("--umap_path", type=str)
    args = parser.parse_args()

    plot_dr(args.adata_path, args.umap_path)