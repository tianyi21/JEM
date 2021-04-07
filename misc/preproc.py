#   -*- coding: utf-8 -*-
#
#   preproc.py
#
#   Created by Tianyi Liu on 2021-04-06 as tianyi
#   Copyright (c) 2021. All Rights Reserved.

"""

"""


import scanpy as sc
import pandas as pd
import numpy as np


def read_10x(path):
    path = './drive/MyDrive/dataset/csc2506/filtered/'
    adata = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
    adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
    adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]

    adata.var_names_make_unique()
    adata.obs['bulk_labels'] = pd.read_csv( path + 'bulk_lables.txt', header=None)[0].values
    sc.pp.log1p(adata, copy=True).write('./zheng17_raw.h5ad')
    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count
    sc.pp.normalize_per_cell(adata)          # normalize with total UMI count per cell
    filter_result = sc.pp.filter_genes_dispersion(
        adata.X, flavor='cell_ranger', n_top_genes=5000, log=False)
    adata = adata[:, filter_result.gene_subset]  # filter genes
    sc.pp.normalize_per_cell(adata)  # need to redo normalization after filtering
    sc.pp.log1p(adata)  # log transform: X = log(X + 1)
    sc.pp.scale(adata)

    orig_label = adata.obs["bulk_labels"].copy()
    unique_label = np.unique(orig_label)
    for i, uniq in enumerate(unique_label):
        orig_label[orig_label == uniq] = i

    adata.obs["int_label"] = orig_label

    return adata
