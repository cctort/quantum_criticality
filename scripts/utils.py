import numpy as np
from h5 import HDFArchive
import itertools
import os
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

def make_hdf_safe(obj):
    
    if obj is None:
        return np.nan
    elif isinstance(obj, bool):
        return int(obj)
    
    elif isinstance(obj, (list, tuple)):
        return type(obj)(make_hdf_safe(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: make_hdf_safe(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return np.array([make_hdf_safe(x) for x in obj], dtype=float)
        else:
            return obj
    else:
        return obj
    
def HDFwrite_dict(filename, results):
    
    with HDFArchive(filename, 'w') as f:

        safe_results = {k: make_hdf_safe(v) for k, v in results.items()}

        # A list for each result
        for key, data in safe_results.items():
            f[key] = data

def HDFwrite_list(filename, results_list):
    
    with HDFArchive(filename, 'w') as f:

        safe_results = [make_hdf_safe(results) for results in results_list]

        f['results_list'] = safe_results

def HDFread_dict(filename, obs_key_list):

    results = {}
    with HDFArchive(filename, 'r') as f:
        
        # Read U,T,n parameters
        results['par_list'] = f['par_list']

        # Read the observables of interest
        for key in obs_key_list:
            results[key] = f[key]
                
    return results

def HDFread_list(filename, obs_key_list):

    obs_key_list.append('par_list')
    results = {key: [] for key in obs_key_list}

    with HDFArchive(filename, 'r') as f:
        results_list = f['results_list']  # this is the stored list of dicts

        for res in results_list:
            for key in obs_key_list:
                if key in res.keys():
                    results[key].append(res[key])
    
    return results

def params_to_dict(params):

    keys = list(params.keys())
    values = [v if isinstance(v, (list, np.ndarray)) else [v] for v in params.values()]
    
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def to_dict_of_lists(params):
    keys = params[0].keys()
    return {k: np.array([d[k] for d in params]) for k in keys}

def make_poly(degrees):
    def poly(x, *params):
        return sum(par * x**deg for par, deg in zip(params, degrees))
    return poly

def sweep_parallel(worker, sweep_list, save_file=None, n_jobs=-1):
    with threadpool_limits(1):
        results_list = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
            delayed(worker)(**{**sweep_dict, 'verbose': False})
            for sweep_dict in sweep_list
        )
    if save_file is not None:
        HDFwrite_list(save_file, results_list)
    return results_list

def critical1(x, a, b, c):
    return a * x ** b - np.sign(c)*a * np.abs(c) **b

def critical2(x, a, b, c):
    return a * np.sign(x - c)*np.abs(x - c)**b