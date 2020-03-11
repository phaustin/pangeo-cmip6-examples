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
# # Load CMIP6 Data with Intake ESM
#
# [Intake ESM](https://intake-esm.readthedocs.io/en/latest/) is an experimental new package that aims to provide a higher-level interface to searching and loading Earth System Model data archives, such as CMIP6. The packages is under very active development, and features may be unstable. Please report any issues or suggestions [on github](https://github.com/NCAR/intake-esm/issues).

# %%
import xarray as xr
xr.set_options(display_style='html')
import intake
# %matplotlib inline

# %% [markdown]
# Intake ESM works by parsing an [ESM Collection Spec](https://github.com/NCAR/esm-collection-spec/) and converting it to an [intake catalog](https://intake.readthedocs.io/en/latest). The collection spec is stored in a .json file. Here we open it using intake.

# %%
cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)
col

# %% [markdown]
# We can now use intake methods to search the collection, and, if desired, export a pandas dataframe.

# %%
cat = col.search(experiment_id=['historical', 'ssp585'], table_id='Oyr', variable_id='o2',
                 grid_label='gn')
cat.df

# %% [markdown]
# Intake knows how to automatically open the datasets using xarray. Furthermore, intake esm contains special logic to concatenate and merge the individual results of our query into larger, more high-level aggregated xarray datasets.

# %%
dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True})
list(dset_dict.keys())

# %%
ds = dset_dict['CMIP.CCCma.CanESM5.historical.Oyr.gn']
ds

# %%
