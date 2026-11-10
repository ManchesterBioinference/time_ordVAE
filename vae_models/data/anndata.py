import torch
import scanpy as sc
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from anndata import AnnData
import scipy

def ann_collate(data):
    """
       data: is a list of (tensor, Dict)

       This is the correct form for the trainer already so no collate is required.
    """

    return data

def ann_time_collate(data):
    """
       data: is a list of (tensor, Dict)

       This is the correct form for the trainer already. Function modifys time column
       to contain integers corresponding to sorted time windows

    """

    #indexer = pd.Index(data[1]['time']).dropna().drop_duplicates().sort_values()
    #data[1]['time'] = torch.tensor(indexer.get_indexer(data[1]['time']))
    return data

class RNA_AnnDataset(Dataset):
    def __init__(self, h5ad: str | AnnData, meta_col_names=None):

        if isinstance(h5ad, str):
            self.file_loc = h5ad
            self.anndata = sc.read_h5ad(self.file_loc)
        elif isinstance(h5ad, AnnData):
            self.anndata = h5ad
            self.file_loc = None
        else:
            raise TypeError('h5ad must be either a string (of file loc) or a AnnData object')

        self.meta_labs_names = meta_col_names

    def __len__(self):
        return self.anndata.shape[0]
    
    def __getitem__(self, idx):
        cell = torch.tensor(self.anndata.X[idx, :], dtype=torch.float)

        if self.meta_labs_names is None:
            return [cell]
        elif isinstance(self.meta_labs_names, dict):

            meta_labels = {}
            for key_lab, val_lab in self.meta_labs_names.items():
                dtype = self._check_metavar_dtype(self.anndata.obs[val_lab])
                meta_labels[key_lab] = torch.reshape(torch.tensor(self.anndata.obs.loc[self.anndata.obs.index[idx],
                                                                                       val_lab].values, dtype=dtype), (-1, 1))
        elif isinstance(self.meta_labs_names, list):
            meta_labels = {}
            for lab in self.meta_labs_names:
                dtype = self._check_metavar_dtype(self.anndata.obs[lab])
                meta_labels[lab] = torch.reshape(torch.tensor(self.anndata.obs.loc[self.anndata.obs.index[idx],
                                                                                   lab].values, dtype=dtype), (-1, 1))
        else:
            raise TypeError('meta_labs_names must be None | list | dict')
            
        return [cell, meta_labels]
    
    def __getitems__(self, idxs):
        if scipy.sparse.issparse(self.anndata.X):
            cells = torch.tensor(self.anndata.X[idxs, :].todense(), dtype=torch.float)
        else:
            cells = torch.tensor(self.anndata.X[idxs, :], dtype=torch.float)

        if self.meta_labs_names is None:
            return [cells]
        elif isinstance(self.meta_labs_names, dict):
            meta_labels = {}
            for key_lab, val_lab in self.meta_labs_names.items():
                dtype = self._check_metavar_dtype(self.anndata.obs[val_lab])
                meta_labels[key_lab] = torch.reshape(torch.tensor(self.anndata.obs.loc[self.anndata.obs.index[idxs],
                                                                                       val_lab].values, dtype=dtype),(-1, 1))
        elif isinstance(self.meta_labs_names, list):
            meta_labels = {}
            for lab in self.meta_labs_names:
                dtype = self._check_metavar_dtype(self.anndata.obs[lab])
                meta_labels[lab] = torch.reshape(torch.tensor(self.anndata.obs.loc[self.anndata.obs.index[idxs],
                                                                                   lab].values, dtype=dtype), (-1, 1))
        else:
            raise TypeError('meta_labs_names must be None | list | dict') 
        
        return [cells, meta_labels]
    
    def _check_metavar_dtype(self, series: pd.Series):
        if series.dtype == int:
            dtype = torch.int
        elif series.dtype == float:
            dtype = torch.float
        else:
            raise TypeError('dtype of meta labels (df columns), must be int or float')
        return dtype
    
    def split(self, proportions:list[int]):
        idx = np.arange(self.anndata.shape[0])
        np.random.shuffle(idx)  #inplace
        n_cells = len(idx)

        cum_props = np.cumsum(proportions)
        cum_props = np.insert(cum_props, 0, 0)
        edges = (n_cells*cum_props).astype(int)


        return [RNA_AnnDataset(self.anndata[idx[start:end]].copy(),
                            meta_col_names=self.meta_labs_names)
                    for start, end in zip(edges[:-1], edges[1:])]
