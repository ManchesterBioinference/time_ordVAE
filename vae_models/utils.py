
from torch import nn
from .types import *
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F


def smooth_log(x, alpha=1e4):
    return torch.log(F.softplus(alpha * x)) - torch.log(torch.tensor(alpha))

def FC_block(first_layer: int,
             last_layer: int,
             hidden_dims: List,
             activations: List | None = None):
    modules = []

    if activations is None:
        activations = (len(hidden_dims) + 1)*[nn.ReLU()]
    
    in_channels = first_layer
    # Build Encoder
    for i, h_dim in enumerate(hidden_dims):
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                activations[i])
        )
        in_channels = h_dim

    modules.append(
        nn.Sequential(
                nn.Linear(in_channels, last_layer),
                nn.BatchNorm1d(last_layer),
                activations[-1]))

    return nn.Sequential(*modules)

def string_to_splits(series: pd.Series, pad_edge=False, inf_edge = False, device='cuda'):
    '''
    Format should be list of  number_number
    '''
    if series.isnull().values.any():
        raise ValueError('nan values in time window col')

    edges = np.array(list(pd.Index(series).drop_duplicates().str.split('_')), dtype='float')
    edges.sort(axis=0)
    
    vals = np.sort(edges.ravel())

    new_edges = []
    i = 0
    for i in range(len(vals)-1):
        if vals[i] == vals[i+1]:
            continue
        new_edges.append((vals[i], vals[i+1]))
    
    new_edges = torch.tensor(new_edges, dtype=torch.float, device=device)
    y_stars = torch.zeros((len(new_edges), len(edges)), device=device)

    for iy, ix in np.ndindex(y_stars.shape):
        y_stars[iy, ix] = 1 if (torch.mean(new_edges[iy]) > edges[ix][0]) & \
                               (torch.mean(new_edges[iy]) < edges[ix][1]) \
                            else 0
    
    mask = ~torch.all(y_stars == 0, axis=1)
    
    y_stars[mask,:] = y_stars[mask]/y_stars[mask].sum(axis=1, keepdims=True)
    
    if pad_edge:
        y_stars = torch.cat((torch.zeros((1, y_stars.shape[1]), device=device),
                               y_stars,
                               torch.zeros((1, y_stars.shape[1]), device=device)), dim=0)
        new_edges = torch.cat((torch.tensor([[-np.inf, new_edges[0,0]]], device=device),
                               new_edges,
                               torch.tensor([[new_edges[-1,-1], np.inf]], device=device)), dim=0)

    else:
        if inf_edge:
            new_edges[[0, -1], [0, -1]] = torch.tensor([-np.inf, np.inf], device=device)

    new_edges_str = ['_'.join(list(row.astype('str'))) for row in np.array(new_edges.cpu())]
    old_edges_str = ['_'.join(list(row.astype('str'))) for row in np.array(edges)]

    y_stars_df = pd.DataFrame(index = new_edges_str, data = np.array(y_stars.cpu()), columns=old_edges_str)
    levels = pd.Index(old_edges_str)  #levels of time

    old_edges_series = np.array(list(pd.Index(series).str.split('_')), dtype='float')
    old_edges_series_str = ['_'.join(list(row.astype('str'))) for row in np.array(old_edges_series)]


    new_edges.to(device)
    y_stars.to(device)
    time_levels = torch.tensor(levels.get_indexer(old_edges_series_str), device=device)
    edges = torch.tensor(edges, device=device)
    return y_stars, new_edges, time_levels, edges, y_stars_df

def string_to_splits_np(series: pd.Series, pad_edge=False, inf_edge = False):
    '''
    Format should be list of  number_number
    '''
    edges = np.array(list(pd.Index(series).dropna().drop_duplicates().sort_values().str.split('_')), dtype=float)
    num_cats = len(edges)
    
    vals = np.sort(edges.ravel())

    new_edges = []
    i = 0
    for i in range(len(vals)-1):
        if vals[i] == vals[i+1]:
            continue
        new_edges.append((vals[i], vals[i+1]))
    
    new_edges = np.array(new_edges)
    y_stars = np.zeros((len(new_edges), len(edges)))

    for iy, ix in np.ndindex(y_stars.shape):
        y_stars[iy, ix] = 1 if (np.mean(new_edges[iy]) > edges[ix][0]) & \
                               (np.mean(new_edges[iy]) < edges[ix][1]) \
                            else 0
    
    mask = ~np.all(y_stars == 0, axis=1)
    y_stars = y_stars[mask]  #drop rows of all 0
    new_edges = new_edges[mask]

    y_stars = y_stars/y_stars.sum(axis=1, keepdims=True)
    if pad_edge:
        new_edges = np.insert(new_edges, [0, len(new_edges)],
                              [[-np.inf, edges[0,0]], [edges[-1,-1], np.inf]], axis=0)
        y_stars = np.insert(y_stars, [0, len(y_stars)], [[0],[0]], axis=0)

    else:
        if inf_edge:
            new_edges[[0, -1], [0, -1]] = np.array([-np.inf, np.inf])

    levels = pd.Index(series).dropna().drop_duplicates().sort_values()  #levels of time 

    return y_stars, new_edges, levels.get_indexer(series)

def str_to_ord_levels(time: pd.Series, pad_edge=False):
    levels = pd.Index(time).dropna().drop_duplicates().sort_values()  #levels of time
    return levels.get_indexer(time)