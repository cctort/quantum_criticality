import numpy as np
from h5 import HDFArchive
import itertools
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import inspect

def merge_results(results, force_array=None):
    merged = {}
    keys = results[0].keys()
    force_array = set(force_array or [])

    for key in keys:
        values = [r[key] for r in results]

        first = values[0]
        same = all(
            np.array_equal(first, v) if isinstance(first, np.ndarray)
            else first == v
            for v in values[1:]
        )

        if same and key not in force_array:
            merged[key] = first
        else:
            try:
                merged[key] = np.asarray(values)
            except Exception:
                merged[key] = values

    return merged

def serialize(obj):

    if obj is None:
        return np.nan

    if inspect.isfunction(obj):
        return obj.__name__

    if isinstance(obj, np.generic):
        return obj.item()

    if hasattr(obj, "to_dict"):
        return serialize(obj.to_dict())

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            sv = serialize(v)

            if isinstance(sv, dict) and not hasattr(v, "to_dict"):
                out.update(sv)
            else:
                out[k] = sv

        return out

    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]

    return obj

def HDFwrite_dict(filename, results, group=None):
    with HDFArchive(filename, 'a') as f:
        target = f
        if group is not None:
            if group not in f:
                f.create_group(group)
            target = f[group]
        safe_results = {k: make_hdf_safe(v) for k, v in results.items()}
        for key, data in safe_results.items():
            try:
                target[key] = data
            except Exception as e:
                print(f"  [HDF SKIP] '{key}': {type(data).__name__} — {e}")

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

def _merge_values(vals):
    """
    Merge a list of values (one per task file) for the same key.
    Recurses into dicts; concatenates arrays/lists; falls back to
    a plain list if the type isn't mergeable.
    """
    first = vals[0]

    if isinstance(first, dict):
        merged = {}
        for k in first.keys():
            merged[k] = _merge_values([v[k] for v in vals])
        return merged

    if isinstance(first, np.ndarray):
        return np.concatenate(vals, axis=0)

    if isinstance(first, list):
        return sum(vals, [])

    # scalar, string, or anything else not naturally concatenable
    # if all tasks agree, just keep one copy; otherwise keep the list
    if all(v == first for v in vals):
        return first
    return vals


def merge_rpa_tasks(out_dir='./data/rpa_cube_clean', filename='test', save=True, verbose=True):
    """
    Merge all per-task HDF5 result files into a single file.
    Walks the full dict structure of each file and merges matching keys
    (concatenating arrays/lists, recursing into nested dicts), with no
    assumptions about what was swept or how files are indexed.
    """
    h5_files = sorted(glob.glob(f'{out_dir}/{filename}_task*.h5'))
    if not h5_files:
        raise FileNotFoundError(f'No task files found matching {filename}_task*.h5 in {out_dir}')

    tasks = []
    for p in h5_files:
        with HDFArchive(p, 'r') as f:
            tasks.append({key: f[key] for key in f.keys()})

    keys = tasks[0].keys()
    for t in tasks[1:]:
        assert t.keys() == keys, 'mismatched keys between task files'

    merged = {key: _merge_values([t[key] for t in tasks]) for key in keys}

    if save:
        merged_path = f'{out_dir}/{filename}_merged.h5'
        with HDFArchive(merged_path, 'w') as ar:
            for key, val in merged.items():
                ar[key] = val
        if verbose:
            print(f'Merged {len(tasks)} task files -> {merged_path}')
    elif verbose:
        print(f'Merged {len(tasks)} task files (not saved, save=False)')

    return merged

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

def run_parallel(worker, inputs, workers=-1):

    with threadpool_limits(1):
        results_list = Parallel(n_jobs=workers, verbose=0, backend='loky')(
            delayed(worker)(**{**input}) for input in inputs)
        
    return results_list

def GL(x, a, b, Xc):
    return a * abs(x - Xc)**b * np.sign(x - Xc)

def HMM(x, a, b, Xc):
    return a * (np.sign(x) * abs(x)**b - np.sign(Xc) * abs(Xc)**b)

def OZ(s, a, b, invxi, s_min):
    x = np.pi * (s - s_min)
    return a/(x**2 + invxi**2 + b*x**3)