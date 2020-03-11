# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMIP6 Precipitation Frequency Analysis Example
#
# This notebook shows an advanced analysis case. The calculation was inspired by [Angie Pendergrass](https://staff.ucar.edu/users/apgrass)’s work on precipitation statistics, as described in the following websites / papers:
# - https://journals.ametsoc.org/doi/full/10.1175/JCLI-D-16-0684.1
# - https://climatedataguide.ucar.edu/climate-data/gpcp-daily-global-precipitation-climatology-project
#
# We use [xhistogram](https://xhistogram.readthedocs.io/) to calculate the distribution of precipitation intensity and its changes in a warming climate.

# %%
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
from tqdm.autonotebook import tqdm

from xhistogram.xarray import histogram

# %matplotlib inline
plt.rcParams['figure.figsize'] = 12, 6
# %config InlineBackend.figure_format = 'retina' 

# %% [markdown]
# We assume this notebook is running in a Pangeo environment with the ability to create [Dask Kubernetes](https://kubernetes.dask.org/en/latest/) distributed clusters for processing. If that's not the case, simply skip the cell below. The analysis will go a lot slower but will hopefully still work.

# %%
from dask.distributed import Client
from dask_kubernetes import KubeCluster

cluster = KubeCluster()
cluster.adapt(minimum=1, maximum=10, interval='2s')
client = Client(cluster)
client

# %% [markdown]
# Here we search for all 3-hourly precipitation fields.

# %%
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df.head()

# %%
df_3hr_pr = df[(df.table_id == '3hr') & (df.variable_id == 'pr')]
len(df_3hr_pr)

# %%
df_3hr_pr.head()

# %%
df_3hr_pr.groupby(['experiment_id', 'source_id'])['zstore'].count()

# %%
run_counts = df_3hr_pr.groupby(['source_id', 'experiment_id'])['zstore'].count()
run_counts

# %%
source_ids = []
experiment_ids = ['historical', 'ssp585']
for name, group in df_3hr_pr.groupby('source_id'):
    if all([expt in group.experiment_id.values
            for expt in experiment_ids]):
        source_ids.append(name)
source_ids


# %%
def load_pr_data(source_id, expt_id):
    """
    Load 3hr precip data for given source and expt ids
    """
    uri = df_3hr_pr[(df_3hr_pr.source_id == source_id) &
                         (df_3hr_pr.experiment_id == expt_id)].zstore.values[0]
    
    gcs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(gcs.get_mapper(uri), consolidated=True)
    return ds


# %%
def precip_hist(ds, nbins=100, pr_log_min=-3, pr_log_max=2):
    """
    Calculate precipitation histogram for a single model. 
    Lazy.
    """
    assert ds.pr.units == 'kg m-2 s-1'
    
    # mm/day
    bins_mm_day = np.hstack([[0], np.logspace(pr_log_min, pr_log_max, nbins)]) 
    bins_kg_m2s = bins_mm_day / (24*60*60)

    pr_hist = histogram(ds.pr, bins=[bins_kg_m2s], dim=['lon']).mean(dim='time')
    
    log_bin_spacing = np.diff(np.log(bins_kg_m2s[1:3])).item()
    pr_hist_norm = 100 * pr_hist / ds.dims['lon'] / log_bin_spacing
    pr_hist_norm.attrs.update({'long_name': 'zonal mean rain frequency',
                               'units': '%/Δln(r)'})
    return pr_hist_norm

def precip_hist_for_expts(dsets, experiment_ids):
    """
    Calculate histogram for a suite of experiments.
    Eager.
    """
    # actual data loading and computations happen in this next line
    pr_hists = [precip_hist(ds).load()
            for ds in [ds_hist, ds_ssp]]
    pr_hist = xr.concat(pr_hists, dim=xr.Variable('experiment_id', experiment_ids))
    return pr_hist


# %%
source_ids

# %%
results = {}
for source_id in tqdm(source_ids):
    # get a 20 year period
    ds_hist = load_pr_data(source_id, 'historical').sel(time=slice('1980', '2000'))
    ds_ssp = load_pr_data(source_id, 'ssp585').sel(time=slice('2080', '2100'))
    pr_hist = precip_hist_for_expts([ds_hist, ds_ssp], experiment_ids)
    results[source_id] = pr_hist


# %%
def plot_precip_changes(pr_hist, vmax=5):
    """
    Visualize the output
    """
    pr_hist_diff = (pr_hist.sel(experiment_id='ssp585') - 
                    pr_hist.sel(experiment_id='historical'))
    pr_hist.sel(experiment_id='historical')[:, 1:].plot.contour(xscale='log', colors='0.5', levels=21)
    pr_hist_diff[:, 1:].plot.contourf(xscale='log', vmax=vmax, levels=21)


# %%
title = 'Change in Zonal Mean Rain Frequency'
for source_id, pr_hist in results.items():
    plt.figure()
    plot_precip_changes(pr_hist)
    plt.title(f'{title}: {source_id}')

# %%

# %%
