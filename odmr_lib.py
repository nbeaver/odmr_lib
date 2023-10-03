#!/usr/bin/env python3
# coding: utf-8

import os.path
import numpy as np
import math
import matplotlib.pyplot as plt

# Function definitions
#
# In alphanumeric order for ease of diffing.

def closest_index(arr, val):
    if val > arr.max() or val < arr.min():
        return None
    index = np.argmin(abs(arr - val)) # TODO: use binary search or something else?
    return index

def dBm_to_mW(power_dBm):
    power_mW = np.power(10,(power_dBm)/10)
    return power_mW

def do_deletions(sweeps, delete):
    if delete is None:
        filtered = sweeps
    else:
        filtered = np.delete(sweeps, obj=delete, axis=0)
    return filtered
    
def estimate_baseline(sweeps, n_points=10):
    avg = sweeps.mean(axis=0)
    N = int(n_points/2)
    first = avg[:N].mean()
    last = avg[-N:].mean()
    baseline = (first + last)/2.
    return baseline


def estimate_baseline_stderr(sweeps, n_points=10):
    avg = sweeps.mean(axis=0)
    N = int(n_points/2)
    first = avg[:N]
    last = avg[-N:]
    combined = np.concatenate([first, last])
    stdev = np.std(combined)
    stderr = stdev/np.sqrt(n_points)
    return stderr


def estimate_contrast(sweeps, baseline=None):
    avg = sweeps.mean(axis=0)
    if baseline is None:
        baseline = estimate_baseline(sweeps)
    contrast = abs(avg.min() - baseline)/baseline
    return contrast


def estimate_contrast_at_f1(sweeps, freq, f1, baseline=None, debug=False):
    avg = sweeps.mean(axis=0)
    assert len(avg) == len(freq)
    if baseline is None:
        baseline = estimate_baseline(sweeps)
    i1 = closest_index(freq, f1)
    if i1 is None:
        return np.nan
    val_at_freq = avg[i1]
    contrast = (baseline - val_at_freq)/baseline
    if debug:
        print("i1 = {}, val_at_freq = {}, baseline = {}".format(i1, val_at_freq, baseline))
    return contrast


def estimate_contrast_stderr(sweeps, n_points=10, baseline=None, baseline_stderr=None):
    if baseline is None:
        baseline = estimate_baseline(sweeps, n_points)
    if baseline_stderr is None:
        baseline_stderr = estimate_baseline_stderr(sweeps, n_points)
    avg = sweeps.mean(axis=0)
    min_val = avg.min()
    min_stderr = estimate_min_stderr(sweeps)
    avg = sweeps.mean(axis=0)
    contrast = estimate_contrast(sweeps, baseline=baseline)
    contrast_stderr = math.hypot(min_val*baseline_stderr/(baseline*baseline), min_stderr)
    return contrast_stderr


def estimate_contrast_stderr_at_f1(sweeps, freq, f1, n_points=10, baseline=None, baseline_stderr=None, debug=False):
    if baseline is None:
        baseline = estimate_baseline(sweeps, n_points)
    if baseline_stderr is None:
        baseline_stderr = estimate_baseline_stderr(sweeps, n_points)
    i1 = closest_index(freq, f1)
    if i1 is None:
        return np.nan
    avg = sweeps.mean(axis=0)
    val_at_f1 = avg[i1]
    stderr_at_f1 = estimate_stderr_at_f1(sweeps, freq, f1)
    avg = sweeps.mean(axis=0)
    if debug:
        print("f1 = {}, i1 = {}, val_at_f1 = {}, stderr_at_f1={}".format(f1, i1, val_at_f1, stderr_at_f1))
    contrast = estimate_contrast_at_f1(sweeps, freq, f1, baseline=baseline, debug=debug)
    contrast_stderr = math.hypot(val_at_f1*baseline_stderr/(baseline*baseline), stderr_at_f1)
    return contrast_stderr


def estimate_dipmin(x, y, x1, y_offset=0):
    i_close = np.searchsorted(x, x1)
    y1 = y[i_close]
    dipmin = y1 + y_offset
    return dipmin


def estimate_min_stderr(sweeps):
    avg = sweeps.mean(axis=0)
    i_slice = np.argmin(avg)
    stderr = estimate_stderr(sweeps, i_slice)
    return stderr


def estimate_stderr(sweeps, i):
    avg = sweeps.mean(axis=0)
    arr_slice = sweeps[:,i]
    stdev = arr_slice.std()
    N = len(arr_slice)
    stderr = stdev/np.sqrt(N)
    return stderr


def estimate_stderr_at_f1(sweeps, freq, f1, ):
    i1 = closest_index(freq, f1)
    if i1 is None:
        return np.nan
    stderr = estimate_stderr(sweeps, i1)
    return stderr


def filter_sweeps(sweeps_raw, use_raw=False, skip_last_sweep=False, delete=None):
    filtered = do_deletions(sweeps_raw, delete)
    if use_raw:
        sweeps = filtered
    else:
        i_max = get_i_slice_end(filtered)
        sweeps = filtered[0:i_max]
    if skip_last_sweep:
        good_sweeps = sweeps[0:-1]
        if i_max == filtered.shape[0]:
            print("warning: scan has probably finished, set skip_last_sweep=False")
    else:
        good_sweeps = sweeps
    return good_sweeps


def fit_n_lorentzians(n, x, y, param_guesses, vary_center=True, vary_bkg=True):
    from lmfit.models import LorentzianModel, ConstantModel
    

    def get_baseline_guess(y, n_points=10):
        N = int(n_points/2)
        first = y[:N].mean()
        last = y[-N:].mean()
        baseline = (first + last)/2.
        return baseline
    
    def get_amplitude_guess(sigma, y_peak, y_background):
        import math
        return math.pi*sigma*(y_peak-y_background)

    # guesses
    guess_c = get_baseline_guess(y)
    
    guess_center = {}
    guess_sigma = {}
    guess_dipmin = {}
    guess_amplitude = {}
    for i in range(n):
        guess_center[i] = param_guesses["l{}_center".format(i)]
        guess_sigma[i] = param_guesses["l{}_sigma".format(i)]
        guess_dipmin[i] = param_guesses["l{}_dipmin".format(i)]
        guess_amplitude[i] = 0.8*get_amplitude_guess(guess_sigma[i], guess_dipmin[i], guess_c)

    background = ConstantModel(prefix="constant_")
    if 'offset' in param_guesses:
        background.set_param_hint('constant_c', value=param_guesses['offset'])
    # params
    params = background.make_params(
        constant_c=guess_c
    )
    dip = {}
    prefix = {}
    params_lorentzian = {}
    for i in range(n):
        prefix[i] = "l{}_".format(i)
        dip[i] = LorentzianModel(prefix=prefix[i])
        dip[i].set_param_hint(
            '{}center'.format(prefix[i]),
            value=guess_center[i]
        )
        dip[i].set_param_hint(
            '{}sigma'.format(prefix[i]),
            value=guess_sigma[i]
        )
        dip[i].set_param_hint(
            '{}amplitude'.format(prefix[i]),
            value=guess_amplitude[i]
        )
        params_lorentzian[i] = dip[i].make_params()
        params.update(params_lorentzian[i])
    
    # constraints
    params["constant_c"].set(
        vary=vary_bkg,
    )
    dx = {}
    for i in range(n):
        dx[i] = guess_sigma[i]/2
        params["l{}_center".format(i)].set(
            vary=vary_center,
        )

    # model
    model = background
    for single_dip in dip.values():
        model += single_dip
    init = model.eval(params, x=x)
    fit_result = model.fit(y, params, x=x)
    return fit_result


def get_arr1d(arr2d):
    """
    If there's only one sweep,
    cast it to a 1D array.
    """
    d1, d2 = arr2d.shape
    if d1 == 1 and d2 > 1:
        arr1d = arr2d.reshape(d2)
    else:
        raise ValueError("cannot cast 2D array with shape '{}'".format(arr2d.shape))
    return arr1d


def get_arr2d(arr):
    """
    If there are multiple sweeps, we need the transpose,
    otherwise leave it alone.
    """
    d1, d2 = arr.shape
    if d1 == 1 and d2 > 1:
        sweeps = arr
    elif d1 > 1 and d2 > 1:
        sweeps = arr.transpose()
    else:
        raise ValueError("cannot interpret array with shape '{}'".format(arr.shape))
    return sweeps


def get_clean_name(filepath_raw):
    basename = os.path.basename(filepath_raw)
    name, ext = os.path.splitext(basename)
    return name


def get_date_created_from_mat_header(header_str):
    # Example:
    # 'MATLAB 5.0 MAT-file Platform: nt, Created on: Thu Mar  2 11:39:05 2023'
    import datetime
    prefix, date_str = header_str.split('Created on: ', maxsplit=1)
    date_val = datetime.datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y")
    return date_val


def get_freq_of_min(sweeps, freq):
    avg = sweeps.mean(axis=0)
    assert len(avg) == len(freq)
    i_min = np.argmin(avg)
    freq_of_min = freq[i_min]
    return freq_of_min


def get_i_slice_end(sweeps):
    """
    For an ODMR scan stopped early,
    there will sometimes be a NaN value.
    The following values and sweeps will be all 1s.
    """
    n_points, n_sweeps = sweeps.shape
    i_max = n_sweeps
    for i, sweep in enumerate(sweeps):
        if any(np.isnan(sweep)):
            i_max = i
            break
        elif all(sweep == np.ones(len(sweep))):
            i_max = i
            break
    return i_max


def get_logical_index(filepath):
    filename = os.path.basename(filepath)
    root, ext = os.path.splitext(filename)
    parts = root.split('_')
    index_str = parts[-1]
    index = int(index_str)
    return index


def get_matfile_key(raw_key):
    # scipy.savemat does not permit keys with leading digits or underscores.
    # see: https://github.com/scipy/scipy/issues/5435
    char1 = raw_key[0]
    if char1 == '_':
        new_key = 'm' + raw_key
    elif char1 in '0123456789':
        new_key = 'm_' + raw_key
    else:
        new_key = raw_key
    # "Field names are restricted to 63 characters"
    max_length = 63
    key = new_key[0:max_length]
    return key


def get_name(filepath):
    filename = os.path.basename(filepath)
    root, ext = os.path.splitext(filename)
    return root


def get_raw(filepath_raw, delete=None):
    norm_raw_unshaped = np.loadtxt(
        filepath_raw,
        comments='#',
        delimiter='\t',
        unpack=False,
    )
    if len(norm_raw_unshaped.shape) == 1:
        # handle single scan case to ensure consistent array shape
        norm_raw = np.reshape(norm_raw_unshaped, (1, norm_raw_unshaped.shape[0]))
    elif len(norm_raw_unshaped.shape) == 2:
        norm_raw = norm_raw_unshaped
    else:
        raise ValueError("file '{}' has invalid shape '{}'".format(filepath_raw, norm_raw.shape))
    norm_raw_nonzero = norm_raw[~np.all(norm_raw == 0, axis=1)]
    if delete is None:
        filtered = norm_raw_nonzero
    else:
        filtered = np.delete(norm_raw_nonzero, obj=delete, axis=0)
    return filtered


def get_required_offset(lines):
    n_lines = len(lines)
    if n_lines < 2:
        # No offset required if nothing to compare.
        return 0.0
    n_pairs = len(lines)-1
    diff_max = np.zeros(n_pairs)
    for i, (line, next_line) in enumerate(zip(lines, lines[1:])):
        diff = next_line - line
        diff_max[i] = diff.max()
    global_max = diff_max.max()
    return global_max


def get_running_avg_stderr(sweeps, index):
    N = sweeps.shape[0]
    vals = sweeps[:,index]
    running_avg = np.zeros(N)
    running_stderr = np.zeros(N)
    for i in range(0,len(vals)):
        n = i+1
        running_avg[i] = vals[0:n].mean()
        stdev = vals[0:n].std()
        running_stderr[i] = stdev/np.sqrt(n)
    return running_avg, running_stderr


def get_step(vals, tolerance=None):
    diff = np.diff(vals)
    steps = np.unique(vals)
    if len(steps) == 1:
        return steps[0]
    elif tolerance is None:
        raise ValueError('non-uniform step')
    else:
        mean = np.mean(diff)
        deviations = diff - mean
        max_deviation = np.max(deviations)
        if max_deviation < tolerance:
            return mean
        else:
            raise ValueError("max deviation > tolerance: {} > {}".format(max_deviation, tolerance))

def get_var_vals(var_name, start, stop, numdivs):
    if var_name == 'itr':
        vals = list(range(numdivs))
    else:
        vals = np.linspace(start, stop, numdivs+1)

def get_yaml_filename(mat_filename):
    root, ext = os.path.splitext(mat_filename)
    yaml_filename = root + '.yaml'
    return yaml_filename

def mW_to_dBm(power_mW):
    power_dBm = 10*np.log10(power_mW)
    return power_dBm

def order_arr2d_by_xy(x, y, arr2d):
    if x[-1] > x[0]:
        # Normal case, increasing x.
        x_ordered = x
        arr2d_ordered_lr = arr2d
    elif x[0] > x[-1]:
        # Reverse case, decreasing x.
        x_ordered = np.flip(x)
        arr2d_ordered_lr = np.fliplr(arr2d)
    else:
        raise ValueError("unorderable: {}, {}".format(x[0], x[-1]))
    if y[-1] > y[0]:
        # Normal case, increasing y.
        y_ordered = y
        arr2d_ordered = arr2d_ordered_lr
    elif y[0] > y[-1]:
        # Reverse case, decreasing y.
        y_ordered = np.flip(y)
        arr2d_ordered = np.flipud(arr2d_ordered_lr)
    else:
        raise ValueError("unorderable: {}, {}".format(y[0], y[-1]))
    return x_ordered, y_ordered, arr2d_ordered

def parse_princeton_mat_file(mat_dict):
    class ODMR:
        # Simple container class.
        def __repr__(self):
            return "ODMR " + str(list(self.__dict__.keys()))
        def __str__(self):
            return "ODMR " + str(list(self.__dict__.keys()))
    odmr = ODMR()
    
    odmr._mat_header = mat_dict['__header__']
    odmr._mat_version = mat_dict['__version__']
    odmr._mat_globals = mat_dict['__globals__']
    
    odmr.xvals = get_arr1d(mat_dict['xvals'])
    try:
        odmr.yvals = get_arr1d(mat_dict['yvals'])
    except KeyError:
        odmr.yvals = None

    odmr.freq = odmr.xvals
    
    odmr.pl = get_arr2d(mat_dict['pl'])
    odmr.norm_raw = odmr.pl
    odmr.sig = get_arr2d(mat_dict['sig'])
    odmr.signal_raw = odmr.sig
    odmr.ref = get_arr2d(mat_dict['ref'])
    odmr.reference_raw = odmr.ref
    
    return odmr

def parse_princeton_val_with_prefix(coefficient, prefix):
    class ValueWithPrefix:
        def __eq__(self, other):
            if self.value == other.value:
                return True
            else:
                return False
        def __str__(self):
            return "ValueWithPrefix {} = {} '{}'".format(self.value, self.coefficient, self.prefix)
        def __repr__(self):
            return "ValueWithPrefix {} = {} '{}'".format(self.value, self.coefficient, self.prefix)
    exponents = {
        'G' : 1e9,
        'M' : 1e6,
        'k' : 1e3,
        '-' : 1e0,
        'm' : 1e-3,
        'u' : 1e-6,
        'n' : 1e-9,
    }
    if prefix not in exponents:
        raise ValueError("unrecognized prefix: '{}''".format(prefix))
    value_object = ValueWithPrefix()
    value_object.value = coefficient * exponents[prefix]
    value_object.coefficient = coefficient
    value_object.prefix = prefix
    value_object.exponent = exponents[prefix]
    return value_object

def plot_fit(info, plot_unc_band=True):
    class FigureInfo:
        pass
    GHz = 1e-9 # Hz to GHz
    fig, ax = plt.subplots(constrained_layout=True)
    if plot_unc_band:
        ax.fill_between(
            info.x*GHz,
            info.fit.best_fit - info.fit_info.fit_unc,
            info.fit.best_fit + info.fit_info.fit_unc,
            color="#ABABAB",
            label='3$\sigma$ uncertainty band'
        )
    ax.plot(
        info.x*GHz,
        info.y,
        '.-',
    );
    ax.plot(
        info.x*GHz,
        info.fit.best_fit,
        'r-',
        label=info.fit_label,
    )
    ax.set_xlabel("microwave frequency [GHz]")
    ax.set_ylabel("normalized fluorescence [dimensionless]")
    ax.legend()
    figinfo = FigureInfo()
    figinfo.fig = fig
    figinfo.ax = ax
    return figinfo


def plot_initial_fit(info):
    GHz = 1e-9 # Hz to GHz
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(info.x*GHz, info.y, '.-');
    ax.plot(info.x*GHz, info.fit.init_fit, 'b-', label="initial fit") # initial fit
    ax.plot(info.x*GHz, info.fit.best_fit, 'r-', label="final fit")
    class FigureInfo:
        pass
    figinfo = FigureInfo()
    figinfo.fig = fig
    figinfo.ax = ax
    return figinfo


def plot_initial_fit_with_n_components(info):
    # TODO: add initial fit components
    GHz = 1e-9 # Hz to GHz
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(info.x*GHz, info.y, '.-');
    ax.plot(info.x*GHz, info.fit.init_fit, 'b-', label="initial fit") # initial fit
    ax.plot(info.x*GHz, info.fit.best_fit, 'r-', label="final fit")
    class FigureInfo:
        pass
    figinfo = FigureInfo()
    figinfo.fig = fig
    figinfo.ax = ax
    return figinfo


def plot_points_vs_sweep(sweeps):
    n_sweeps, n_points = sweeps.shape
    sweeps_avg = sweeps.mean(axis=0)
    N_avg = [1 + x for x in range(n_sweeps)]
    first, first_stderr = get_running_avg_stderr(sweeps, 0)
    last, last_stderr = get_running_avg_stderr(sweeps, -1)
    minim, minim_stderr = get_running_avg_stderr(sweeps, np.argmin(sweeps_avg))
    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(N_avg, first, fmt='.', yerr=first_stderr, label="first");
    ax.errorbar(N_avg, last, fmt='.', yerr=last_stderr, label="last");
    ax.errorbar(N_avg, minim, fmt='.', yerr=minim_stderr, label="minimum");
    ax.set_xlabel("number of sweeps")
    return fig, ax


def plot_residuals(info, n_stderr=1):
    fig, ax = plt.subplots(constrained_layout=True)
    GHz = 1e-9 # Hz to GHz
    ax.plot(info.x*GHz, info.y_residuals, '.-', label="residuals")
    ax.scatter(info.x*GHz, info.y_stderr, marker='.', color="black", label="stderr");
    ax.scatter(info.x*GHz, -info.y_stderr, marker='.', color="black");
    if n_stderr == 2:
        ax.scatter(info.x*GHz, 2*info.y_stderr, marker='.', color="red", label="2*stderr");
        ax.scatter(info.x*GHz, -2*info.y_stderr, marker='.', color="red");
    elif n_stderr not in [1, 2]:
        raise ValueError
    class FigureInfo:
        pass
    figinfo = FigureInfo()
    figinfo.fig = fig
    figinfo.ax = ax
    return figinfo


def plot_n_components(info, n=None):
    if n is None:
        N_lorentz = info.n_lorentzians
    else:
        N_lorentz = n
    GHz = 1e-9 # Hz to GHz
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(info.x*GHz, info.y, '.-');
    y0 = info.fit_components['constant_']
    ax.axhline(y0, linestyle='--', color="gray", label="constant")
    for i in range(N_lorentz):
        ax.plot(
            info.x*GHz,
            info.fit_components['l{}_'.format(i)] + y0,
            label="{}".format(i))
    ax.plot(info.x*GHz, info.fit.best_fit, 'r-', label="final fit")
    class FigureInfo:
        pass
    figinfo = FigureInfo()
    figinfo.fig = fig
    figinfo.ax = ax
    return figinfo


def reduce_identical_vals(d):
    """
    Return the first value of a dict
    provided all the values are equal.
    """
    val1 = next(iter(d.values()))
    all_same = all([d[key] == val1 for key in d.keys()])
    if all_same == True:
        return val1
    else:
        raise ValueError("dict has disparate values")


def reduce_identical_array_vals(d):
    """
    Return the first value of a dict
    provided all the values are equal
    and are numpy arrays.
    """
    val1 = next(iter(d.values()))
    all_same = all([np.array_equal(d[key], val1) for key in d.keys()])
    if all_same == True:
        return val1
    else:
        raise ValueError("dict has disparate values")

def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    else:
        return text


def yield_odmr_mat_paths(folder, pattern="PLmw1freq*.mat"):
    import glob
    odmr_pattern = os.path.join(folder, pattern)
    paths = glob.glob(odmr_pattern)
    for path in paths:
        yield path

