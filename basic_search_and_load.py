# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info,-toc,-latex_envs
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Google Cloud CMIP6 Public Data: Basic Python Example
#
# This notebooks shows how to query the catalog and load the data using python

# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import gcsfs

xr.set_options(display_style='html')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' 

# %%
plt.rcParams['figure.figsize'] = 12, 6

# %% [markdown]
# ## Browse Catalog
#
# The data catatalog is stored as a CSV file. Here we read it with Pandas.

# %%
df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
df.head()

# %% [markdown]
# The columns of the dataframe correspond to the CMI6 controlled vocabulary. A beginners' guide to these terms is available in [this document](https://docs.google.com/document/d/1yUx6jr9EdedCOLd--CPdTfGDwEwzPpCF6p1jRmqx-0Q). 
#
# Here we filter the data to find monthly surface air temperature for historical experiments.

# %%
df_ta = df.query("activity_id=='CMIP' & table_id == 'Amon' & variable_id == 'tas' & experiment_id == 'historical'")
df_ta

# %% [markdown]
# Now we do further filtering to find just the models from NCAR.

# %%
df_ta_ncar = df_ta.query('institution_id == "NCAR"')
df_ta_ncar

# %% [markdown]
# ## Load Data
#
# Now we will load a single store using gcsfs, zarr, and xarray.

# %%
# this only needs to be created once
gcs = gcsfs.GCSFileSystem(token='anon')

# get the path to a specific zarr store (the first one from the dataframe above)
zstore = df_ta_ncar.zstore.values[-1]

# create a mutable-mapping-style interface to the store
mapper = gcs.get_mapper(zstore)

# open it using xarray and zarr
ds = xr.open_zarr(mapper, consolidated=True)
ds

# %% [markdown]
# Plot a map from a specific date.

# %%
ds.tas.sel(time='1950-01').squeeze().plot()

# %% [markdown]
# Create a timeseries of global-average surface air temperature. For this we need the area weighting factor for each gridpoint.

# %%
df_area = df.query("variable_id == 'areacella' & source_id == 'CESM2'")
ds_area = xr.open_zarr(gcs.get_mapper(df_area.zstore.values[0]), consolidated=True)
ds_area

# %%
total_area = ds_area.areacella.sum(dim=['lon', 'lat'])
ta_timeseries = (ds.tas * ds_area.areacella).sum(dim=['lon', 'lat']) / total_area
ta_timeseries

# %% [markdown]
# By default the data are loaded lazily, as Dask arrays. Here we trigger computation explicitly.

# %%
# %time ta_timeseries.load()

# %%
ta_timeseries.plot(label='monthly')
ta_timeseries.rolling(time=12).mean().plot(label='12 month rolling mean')
plt.legend()
plt.title('Global Mean Surface Air Temperature')

# %%
