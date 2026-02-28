import numpy as np
import pandas as pd
from scipy.stats import poisson
import dask.dataframe as dd
import tqdm 
from dask.diagnostics import ProgressBar
import datetime
import os
from covid_constants_and_util import PATH_TO_ACS_5YR_DATA,PATH_TO_NYT_DATA, PATH_TO_ITDPC_DATA, BASE_DIR

###################################################
# Loss functions
###################################################
def MRE(y_true, y_pred):
    '''
    Computes the median relative error (MRE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the MRE is returned.
    If y_true and y_pred are 2D, e.g., predictions over multiple seeds,
    the MRE is computed per row, then averaged.
    '''
    abs_err = np.absolute(y_true - y_pred)
    rel_err = abs_err / y_true
    if len(abs_err.shape) == 1:  # this implies y_true and y_pred are 1D
        mre = np.median(rel_err)
    else:  # this implies at least one of them is 2D
        mre = np.mean(np.median(rel_err, axis=1))
    return mre

def RMSE(y_true, y_pred):
    '''
    Computes the root mean squared error (RMSE). y_true and y_pred should
    both be numpy arrays.
    If y_true and y_pred are 1D, the RMSE is returned.
    If y_true and y_pred are 2D, e.g., predictions over multiple seeds,
    the RMSE is computed per row, then averaged.
    '''
    sq_err = (y_true - y_pred) ** 2
    if len(sq_err.shape) == 1:  # this implies y_true and y_pred are 1D
        rmse = np.sqrt(np.mean(sq_err))
    else:  # this implies at least one of them is 2D
        rmse = np.sqrt(np.mean(sq_err, axis=1))
        rmse = np.mean(rmse)
    return rmse

def MSE(y_true, y_pred):
    '''
    Computes the mean squared error (MSE). y_true and y_pred should
    both be numpy arrays.
    '''
    return np.mean((y_true - y_pred) ** 2)

def poisson_NLL(y_true, y_pred, sum_or_logsumexp):
    # We clip variance to a min of 4, similar to Li et al. (2020)
    # First sum log-likelihoods over days
    variance = np.clip(y_pred, 4, None)
    ll = np.sum(poisson.logpmf(y_true, variance), axis=1)
    # Then sum or logsumexp over seeds
    ll = sum_or_logsumexp(ll)
    return -ll

###################################################
# Helper functions
###################################################

def get_datetime_hour_as_string(datetime_hour):
    return '%i.%i.%i.%i' % (datetime_hour.year, datetime_hour.month,
                            datetime_hour.day, datetime_hour.hour)

def load_csv_possibly_with_dask(filenames, use_dask=False, compression='gzip', blocksize=None, compute_with_dask=True, **kwargs):
    # Avoid loading the index column because it's probably not desired.
    if not ('usecols' in kwargs and kwargs['usecols'] is not None):
        kwargs['usecols'] = lambda col: col != 'Unnamed: 0'
    if use_dask:
        with ProgressBar():
            d = dd.read_csv(filenames, compression=compression, blocksize=blocksize, **kwargs)
            if compute_with_dask:
                d = d.compute()
                d.index = range(len(d))
            return d
    else:
        # Use tqdm to display a progress bar.
        #return pd.concat(pd.read_csv(f, **kwargs) for f in tqdm_wrap(filenames))
        return pd.concat(pd.read_csv(f, **kwargs) for f in tqdm.tqdm(filenames))

def get_cumulative(x):
    '''
    Converts an array of values into its cumulative form,
    i.e. cumulative_x[i] = x[0] + x[1] + ... + x[i]

    x should either be a 1D or 2D numpy array. If x is 2D,
    the cumulative form of each row is returned.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        cumulative_x = []
        curr_sum = 0
        for val in x:
            curr_sum = curr_sum + val
            cumulative_x.append(curr_sum)
        cumulative_x = np.array(cumulative_x)
    else:
        num_seeds, num_time = x.shape
        cumulative_x = []
        curr_sum = np.zeros(num_seeds)
        for i in range(num_time):
            curr_sum = curr_sum + x[:, i]
            cumulative_x.append(curr_sum.copy())
        cumulative_x = np.array(cumulative_x).T
    return cumulative_x

def get_daily_from_cumulative(x):
    '''
    Converts an array of values from its cumulative form
    back into its original form.

    x should either be a 1D or 2D numpy array.
    '''
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 1:
        arr_to_return = np.array([x[0]] + list(x[1:] - x[:-1]))
    else:
        # seeds are axis 0, so want to subtract along axis 1.
        x0 = x[:, :1]
        increments = x[:, 1:] - x[:, :-1]
        arr_to_return = np.concatenate((x0, increments), axis=1)
    if not (arr_to_return >= 0).all():
        bad_val_frac = (arr_to_return < 0).mean()
        print("Warning: fraction %2.3f of values are not greater than 0! clipping to 0" % bad_val_frac)
        print(arr_to_return)
        assert bad_val_frac < 0.1 # this happens quite occasionally in NYT data.
        arr_to_return = np.clip(arr_to_return, 0, None)
    return arr_to_return

def mean_and_CIs_of_timeseries_matrix(M, alpha=0.05):
    """
    Given a matrix which is N_SEEDS X T, return mean and upper and lower CI for plotting.
    """
    assert alpha > 0
    assert alpha < 1
    mean = np.mean(M, axis=0)
    lower_CI = np.percentile(M, 100 * alpha/2, axis=0)
    upper_CI = np.percentile(M, 100 * (1 - alpha/2), axis=0)
    return mean, lower_CI, upper_CI

def apply_smoothing(x, agg_func=np.mean, before=2, after=2):
    new_x = []
    for i, x_point in enumerate(x):
        before_idx = max(0, i-before)
        after_idx = min(len(x), i+after+1)
        new_x.append(agg_func(x[before_idx:after_idx]))
    return np.array(new_x)

# inspired by https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
def reformat_large_tick_values(tick_val, pos):
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        postfix = 'B'
    elif tick_val >= 10000000:  # if over 10M, don't include decimal
        val = int(round(tick_val/1000000, 0))
        postfix = 'M'
    elif tick_val >= 1000000:  # if 1M-10M, include decimal
        val = round(tick_val/1000000, 1)
        postfix = 'M'
    elif tick_val >= 1000:
        val = int(round(tick_val/1000, 0))
        postfix = 'k'
    else:
        val = int(tick_val)
        postfix = ''
    new_tick_format = '%s%s' % (val, postfix)
    return new_tick_format

def reformat_decimal_as_percent(tick_val, pos):
    percent = round(tick_val * 100, 1)
    new_tick_format = '%d%%' % percent
    return new_tick_format

def failsafe_int_conversion(x):
    try:
        return np.int64(x)
    except ValueError:
        try:
            return np.int64(x.split(":")[-1])
        except Exception:
            print(x)
            return np.nan

def match_msa_name_to_msas_in_acs_data(msa_name, acs_msas):
    '''
    Matches the MSA name from our annotated SafeGraph data to the
    MSA name in the external datasource in MSA_COUNTY_MAPPING
    '''
    msa_pieces = msa_name.split('_')
    query_states = set()
    i = len(msa_pieces) - 1
    while True:
        piece = msa_pieces[i]
        if len(piece) == 2 and piece.upper() == piece:
            query_states.add(piece)
            i -= 1
        else:
            break
    query_cities = set(msa_pieces[:i+1])

    for msa in acs_msas:
        if ', ' in msa:
            city_string, state_string = msa.split(', ')
            states = set(state_string.split('-'))
            if states == query_states:
                cities = city_string.split('-')
                overlap = set(cities).intersection(query_cities)
                if len(overlap) > 0:  # same states and at least one city matched
                    return msa
    return None

def get_fips_codes_from_state_and_county_fp(state_vec, county_vec):
    fips_codes = []
    for state, county in zip(state_vec, county_vec):
        state = str(state)
        if len(state) == 1:
            state = '0' + state
        county = str(county)
        if len(county) == 1:
            county = '00' + county
        elif len(county) == 2:
            county = '0' + county
        fips_codes.append(np.int64(state + county))
    return fips_codes

def get_nyt_outcomes_over_counties(counties=None):
    outcomes = pd.read_csv(PATH_TO_NYT_DATA)
    if counties is not None:
        outcomes = outcomes[outcomes['fips'].isin(counties)]
    return outcomes

def get_variables_for_evaluating_msa_model(msa_name, verbose=False):
    acs_data = pd.read_csv(PATH_TO_ACS_5YR_DATA)
    acs_msas = [msa for msa in acs_data['CBSA Title'].unique() if type(msa) == str]
    msa_match = match_msa_name_to_msas_in_acs_data(msa_name, acs_msas)
    if verbose: print('Found MSA %s in ACS 5-year data' % msa_match)

    msa_data = acs_data[acs_data['CBSA Title'] == msa_match].copy()
    msa_data['id_to_match_to_safegraph_data'] = msa_data['GEOID'].map(lambda x:x.split("US")[1]).astype(np.int64)
    msa_cbgs = msa_data['id_to_match_to_safegraph_data'].values
    msa_data['fips'] = get_fips_codes_from_state_and_county_fp(msa_data.STATEFP, msa_data.COUNTYFP)
    msa_counties = list(set(msa_data['fips'].values))
    if verbose:
        print('Found %d counties and %d CBGs in MSA' % (len(msa_counties), len(msa_cbgs)))

    nyt_outcomes = get_nyt_outcomes_over_counties(msa_counties)
    nyt_counties = set(nyt_outcomes.fips.unique())
    nyt_cbgs = msa_data[msa_data['fips'].isin(nyt_counties)]['id_to_match_to_safegraph_data'].values
    if verbose:
        print('Found NYT data matching %d counties and %d CBGs' % (len(nyt_counties), len(nyt_cbgs)))
    return nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs


def get_province_outcomes(msa=None):
    italian_data = pd.read_csv(
        PATH_TO_ITDPC_DATA, 
        usecols=['data','denominazione_provincia','totale_casi'],
        parse_dates=['data'])
    italian_data['data'] = italian_data['data'].dt.strftime("%Y-%m-%d")
    # italian_data = italian_data[pd.isna(italian_data['sigla_provincia'])==False]
    # italian_data = italian_data.drop('sigla_provincia',1)
    
    # no info on deaths
    italian_data.columns = ['date','province','cases']    
    if msa is not None:
        italian_data = italian_data[italian_data['province'] == msa]
    return italian_data

def get_variables_for_evaluating_province_model(msa_name):
    nyt_outcomes = get_province_outcomes(msa_name)
    return nyt_outcomes

def get_dates_by_phase(phase, year=2020):
    MIN_DATETIME = None
    MAX_DATETIME = None
    if (phase == 1) or (phase == 'first'):
        MIN_DATETIME = datetime.datetime(year, 2, 7, 0)
        MAX_DATETIME = datetime.datetime(year, 6, 2, 23)
    elif (phase == 2) or (phase == 'second'):
        #MIN_DATETIME = datetime.datetime(year, 8, 13, 0)
        #MAX_DATETIME = datetime.datetime(year, 12, 30, 23)    
        
        MIN_DATETIME = datetime.datetime(year, 10, 9, 0)
        MAX_DATETIME = datetime.datetime(year, 12, 7, 23)

        #MIN_DATETIME = datetime.datetime(year, 9, 24, 0)        
        # MIN_DATETIME = datetime.datetime(year, 9, 10, 0)
        # MAX_DATETIME = datetime.datetime(year, 12, 7, 23)   
    return MIN_DATETIME,MAX_DATETIME

def define_global_variables_by_phase(wave='first',data_type='filter_rs'):
    global MIN_DATETIME
    global TRAIN_TEST_PARTITION
    global MAX_DATETIME    
    global STRATIFIED_BY_AREA_DIR
    global PATH_TO_IPF_OUTPUT
    global PATH_TO_SAVED_CHARACTERISTICS     

    if wave == "first":
        MIN_DATETIME = datetime.datetime(2020, 2, 7, 0)
        TRAIN_TEST_PARTITION = datetime.datetime(2020, 5, 4)
        MAX_DATETIME = datetime.datetime(2020, 6, 2, 23)    
        path = "first"
    elif wave == "second":
        #TRAIN_TEST_PARTITION = datetime.datetime(2020, 11, 2)
        #MIN_DATETIME = datetime.datetime(2020, 8, 13, 0)
        #MAX_DATETIME = datetime.datetime(2020, 12, 30, 23)

        TRAIN_TEST_PARTITION = datetime.datetime(2020, 11, 6)
        
        MIN_DATETIME = datetime.datetime(2020, 10, 9, 0)
        #MIN_DATETIME = datetime.datetime(2020, 9, 10, 0)
        MAX_DATETIME = datetime.datetime(2020, 12, 7, 23)

        path = "second"
        
    STRATIFIED_BY_AREA_DIR = os.path.join(
        BASE_DIR,    
        'all_aggregate_data',   
        'chunks_with_demographic_annotations_stratified_by_area', 
        data_type, path)
    
    PATH_TO_IPF_OUTPUT = os.path.join(
        BASE_DIR,
        'all_aggregate_data',
        'ipf_output', 
        data_type, path)
    
    PATH_TO_SAVED_CHARACTERISTICS = os.path.join(
        BASE_DIR, 'all_aggregate_data',
        'poi_metadata',
        data_type, path)
    return