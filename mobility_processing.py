import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime
from scipy import sparse
import pickle

BASE_DIR = 'data'    

from covid_constants_and_util import PATH_TO_IPF_OUTPUT, PATH_TO_SAVED_CHARACTERISTICS
from utilities import get_dates_by_phase

def expand_daily_to_hourly(df, col_name='NR_OPERAZIONE'):
    hourly_df = df[col_name].apply(get_24h_values_split).apply(pd.Series)
    expanded_df = pd.concat([df,hourly_df], axis=1)
    # check before dropping that the sum of the new columns corresponds
    # to the original column
    try:
        #this may fail because of float precision rounding
        assert (expanded_df[col_name].fillna(0) == expanded_df[range(24)].sum(1)).all(), f"Sum test failed for col {col_name}"
    except AssertionError: 
        #this may allow some errors, which we constrain to be small on average
        assert (expanded_df[col_name].fillna(0) - expanded_df[range(24)].sum(1)).mean() < 10**-5, f"Mean difference test failed for value {col_name}"
    
    expanded_df = expanded_df.drop(col_name,1)
    return expanded_df


def get_ipf_filename(msa_name, min_datetime, max_datetime, clip_visits, correct_visits=True, output_path=PATH_TO_IPF_OUTPUT):
    """
    Get the filename matching these parameters of IPF.
    """
    fn = '%s_%s_to_%s_clip_visits_%s' % (
        msa_name,
        min_datetime.strftime('%Y-%m-%d'),
        max_datetime.strftime('%Y-%m-%d'),
        clip_visits)
    if correct_visits:
        fn += '_correct_visits_True'
    filename = os.path.join(output_path, '%s.pkl' % fn)
    return filename

def get_duplicate_municipalities_df():
    duplicate_muns = """nome;nome_provincia;sigla;nome_disamb
    Calliano;Asti;AT;Calliano (AT)
    Calliano;Trento;TN;Calliano (TN)
    Castro;Bergamo;BG;Castro (BG)
    Castro;Lecce;LE;Castro (LE)
    Livo;Como;CO;Livo (CO)
    Livo;Trento;TN;Livo (TN)
    Peglio;Como;CO;Peglio (CO)
    Peglio;Pesaro e Urbino;PU;Peglio (PU)
    Samone;Torino;TO;Samone (TO)
    Samone;Trento;TN;Samone (TN)
    San Teodoro;Messina;ME;San Teodoro (ME)
    San Teodoro;Sassari;SS;San Teodoro (SS)""".split("\n")

    duplicate_muns_df = pd.DataFrame(
        map(lambda x: x.split(";"), duplicate_muns[1:]), 
        columns=duplicate_muns[0].split(";"))
    return duplicate_muns_df

def get_24h_values_split(x):
    """    
    Split the sample in 3 unequal parts:
    - 00:00-08:00 10% of observations
    - 08:00-16:00 60% of observations
    - 16:00-24:00 30% of observations
    """    
    if np.isnan(x) or x == 0:
        return np.zeros(24)
    p10,p70 = np.percentile(np.arange(x+1),[10,70])
    # get cumulative sum of values
    new_range = np.concatenate([
        np.linspace(0,p10,9, endpoint=False), #UGLY, probably wrong?
        np.linspace(p10,p70,8, endpoint=False), 
        np.linspace(p70,x,8)])
    # from cumulative sum obtain single values 
    new_range[1:] -= new_range[:-1].copy()
    # remove 0 value at start
    new_range = new_range[1:]
    
    try:
        #this may fail because of float precision rounding
        assert new_range.sum()==x, f"Sum test failed for value {x}"
    except AssertionError: 
        #this may allow some errors, which we constrain to be small on average
        assert np.mean(new_range.sum()-x) < 10**-5, f"Mean difference test failed for value {x}"
    return new_range

def export_all_hourly_matrices(data, t_start, t_end, filter_row_by=None, verbose=False): 
    if type(t_start) == datetime.datetime:
        t_start = datetime.datetime.strftime(t_start,"%Y-%m-%d")   
    
    if type(t_end) == datetime.datetime:
        t_end = datetime.datetime.strftime(t_end,"%Y-%m-%d")   
    assert t_start != t_end
    
    t_start = data.columns.tolist().index(t_start)
    t_end = data.columns.tolist().index(t_end)
    days_to_load = data.columns[t_start:(t_end+1)].tolist()

    if filter_row_by is not None:
        data = data[data['nome_provincia_client'].isin(filter_row_by)]
    
    hourly_matrices = []  
    ref_index = None  
    ref_cols = None  
    for day in days_to_load:    
        hourly_data = pd.DataFrame(
            data[day].apply(get_24h_values_split).tolist(), 
            index=data.index)
        
        hourly_data = hourly_data.reset_index()
        for h in range(24):
            hourly_data_bipartite = hourly_data.pivot(
                index='nome_client', 
                columns='unique_id_vendor', values=h)
            
            if ref_index is None:
                print("Reference index missing: storing it.")
                ref_index = hourly_data_bipartite.index.tolist()
            else:
                assert ref_index == hourly_data_bipartite.index.tolist()

            if ref_cols is None:
                print("Reference column list missing: storing it.")
                ref_cols = hourly_data_bipartite.columns.tolist()
            else:
                assert ref_cols == hourly_data_bipartite.columns.tolist()
            
            hourly_data_bipartite = hourly_data_bipartite.fillna(0)
            m = sparse.csr_matrix(hourly_data_bipartite.values)
            hourly_matrices.append(m)
        # print every 7 days
        if verbose and (days_to_load.index(day)%7 == 0 or days_to_load.index(day) == 0):       
            print(f"{datetime.datetime.now().time()} - Export hourly matrix of day {day}")
    return hourly_matrices, ref_index, ref_cols

def get_metadata_filename(
    msa_name, min_datetime, max_datetime, clip_visits, 
    correct_visits=True, output_path=PATH_TO_SAVED_CHARACTERISTICS):
    """
    Get the filename matching these parameters of IPF.
    """
    fn = '%s_%s_to_%s_clip_visits_%s' % (
        msa_name,
        min_datetime.strftime('%Y-%m-%d'),
        max_datetime.strftime('%Y-%m-%d'),
        clip_visits)
    if correct_visits:
        fn += '_correct_visits_True'
    filename = os.path.join(output_path, 'metadata_%s.pkl' % fn)
    return filename

# def get_dates_by_phase(phase, year=2020):
#     if phase == 1:
#         MIN_DATETIME = datetime.datetime(year, 2, 7, 0)
#         MAX_DATETIME = datetime.datetime(year, 6, 2, 23)
#     elif phase == 2:
#         MIN_DATETIME = datetime.datetime(year, 8, 13, 0)
#         MAX_DATETIME = datetime.datetime(year, 12, 30, 23)    
#     return MIN_DATETIME,MAX_DATETIME

def manage_exporting_paths(phase,filter_to_use, year=2020):
    
    #MIN_DATETIME,MAX_DATETIME = get_dates_by_phase(phase, year=year)
    #fn = f"{MIN_DATETIME.strftime('%Y-%m-%d')}_to_{MAX_DATETIME.strftime('%Y-%m-%d')}" 
    fn = ""  
    if year == 2020:
        if (phase == 1) or (phase == 'first'):
            fn = 'first'
        elif (phase == 2) or (phase == 'second'):
            fn = 'second' 
    
    if year == 2019:
        if (phase == 1) or (phase == 'first'):
            fn = 'first_2019'
        elif (phase == 2) or (phase == 'second'):
            fn = 'second_2019' 
    
    # NEW_rebuttal_20230106
    if year == 2021:
        if (phase == 1) or (phase == 'first'):
            fn = 'first_2021'
        elif (phase == 2) or (phase == 'second'):
            fn = 'second_2021' 

    PATH_TO_IPF_OUTPUT = os.path.join(
        'data','all_aggregate_data','ipf_output', 
        filter_to_use, fn)
    STRATIFIED_BY_AREA_DIR = os.path.join(
        BASE_DIR, 
        'all_aggregate_data',
        'chunks_with_demographic_annotations_stratified_by_area', 
        filter_to_use, fn)
    PATH_TO_SAVED_CHARACTERISTICS = os.path.join(
        BASE_DIR, 
        'all_aggregate_data',
        'poi_metadata', filter_to_use, fn)

    for folder in [
        PATH_TO_IPF_OUTPUT,
        STRATIFIED_BY_AREA_DIR,
        PATH_TO_SAVED_CHARACTERISTICS]:
        try:
            assert os.path.exists(folder)
        except AssertionError:
            print("Output folder not found, making it:")
            print(folder)
            os.makedirs(folder)
               
    return PATH_TO_IPF_OUTPUT,STRATIFIED_BY_AREA_DIR,PATH_TO_SAVED_CHARACTERISTICS
    
def get_regional_data_path(region,filter_to_use):
    return os.path.join(
        f'italian_data_{filter_to_use}',
        'monthly_edgelists',
        'by_region',region)

def get_geographic_details(top_region):
    ref_df = pd.read_csv("italian_data/cap_comuni.csv")
    provinces = ref_df[['nome_regione','nome_provincia']].drop_duplicates()
    population_data = ref_df[['nome','popolazione','nome_provincia']]
    reg_ref_dict = provinces.set_index('nome_provincia')['nome_regione'].to_dict()
    top_region_dict = {
        r:[k for k,v in reg_ref_dict.items() if v==r] 
        for r in top_region
    }    
    return provinces, population_data, top_region_dict    
    
def get_hourly_mobility_matrices(region, filter_to_use, phase, year=2020):    
    duplicate_muns_df = get_duplicate_municipalities_df()
    regional_data_path = get_regional_data_path(region, filter_to_use)
    
    provinces, _, top_region_dict = get_geographic_details([region])
    provinces_top = top_region_dict[region]
    
    MIN_DATETIME,MAX_DATETIME = get_dates_by_phase(phase, year=year)
    
    # load all data from provinces in the same region    
    province_data = []    
    for province in provinces_top:    
        # load province data
        province_mobility_data = pd.read_csv(
            os.path.join(regional_data_path,f'{province}_{year}.csv'), 
            usecols=lambda x: x not in ['nome_provincia'])

        # remove duplicate municipalities inside the data
        for i,r in duplicate_muns_df.iterrows():
            mun,prov,_,new_mun = r
            province_mobility_data.loc[
                (province_mobility_data['nome_client'] == mun) & 
                (province_mobility_data['nome_provincia_client'] == prov), 
                'nome_client'] = new_mun
                
        # reformat data
        province_mobility_data = province_mobility_data.set_index([
            'nome_client','unique_id_vendor'])    
        
        if filter_to_use == 'filter5_rs':
            for c in province_mobility_data.columns:
                if c != 'nome_provincia_client':
                    province_mobility_data.loc[province_mobility_data[c] < 5, c] = np.nan
        
        province_data.append(province_mobility_data)       

    # check that new method is equivalent to old one
    regional_provinces = provinces.loc[
        provinces['nome_regione']==region]['nome_provincia'].tolist()  
    assert set(regional_provinces) == set(provinces_top)   
    
    # check that columns are the same across all dfs
    for df in province_data:
        assert set(df.columns) == set(province_data[0].columns)
    
    region_mobility_data = pd.concat(province_data)
    
    hourly_matrices, ref_index, ref_column = export_all_hourly_matrices(
        data=region_mobility_data, 
        t_start=MIN_DATETIME, 
        t_end=MAX_DATETIME, 
        filter_row_by=provinces_top, 
        verbose=True)    
    
    return hourly_matrices, ref_index, ref_column

def get_population_data(provinces_top, population_data, ref_index, provinces_to_track=None):
    # obtain population data for the full region
    # obtain population data from the province in the region
    prov_population_data = population_data[
        population_data['nome_provincia'].isin(provinces_top)]

    # remove duplicates
    duplicate_muns_df = get_duplicate_municipalities_df()
    for i,r in duplicate_muns_df.iterrows():
        mun,prov,_,new_mun = r
        prov_population_data.loc[
            (prov_population_data['nome'] == mun) & 
            (prov_population_data['nome_provincia'] == prov), 'nome'] = new_mun

    # obtain population data from the municipalities in the target province
    # this duplicates the data but it is for 
    province_mun = set(prov_population_data[
        prov_population_data['nome_provincia'].isin(provinces_top)]['nome'].unique())

    # format data
    prov_population_data_s = prov_population_data.drop('nome_provincia',1)
    # this is needed because the data is at the CAP level so we have duplicates
    prov_population_data_s = prov_population_data_s.drop_duplicates()
    prov_population_data_s = prov_population_data_s.groupby(['nome']).sum()
    # this is needed to ensure that we have the same order in the hourly matrices and the population data
    prov_population_data_s = prov_population_data_s.reindex(ref_index)
    prov_population_data_s = prov_population_data_s.reset_index()
    prov_population_data_s.columns = ['municipality','population']
    prov_population_data_s['track'] = prov_population_data_s['municipality'].isin(
        province_mun).astype(int)

    for p in provinces_to_track:
        province_mun = set(prov_population_data[
            prov_population_data['nome_provincia'] == p]['nome'].unique())            
        prov_population_data_s[p] = prov_population_data_s['municipality'].isin(
            province_mun).astype(int)    
    return prov_population_data_s

def get_metadata_to_export(region,ref_index,ref_column):
    r_metadata = {
        'municipalities': ref_index, 
        'POIs': ref_column}          
    metadata_to_export = {region : {}}
    metadata_to_export[region]['municipalities'] = r_metadata['municipalities']
    metadata_to_export[region]['POI_municipalities'] = list(
        map(lambda x: " ".join(x.split("_")[:-1]), r_metadata['POIs']))
    metadata_to_export[region]['POI_categories'] = list(
        map(lambda x: x.split("_")[-1], r_metadata['POIs']))
    return metadata_to_export

def threshold_filtering_test(poi_cbg_visits_list):
    day_indices = range(0,len(poi_cbg_visits_list),24)
    daily_tests = []
    for i,d in enumerate(day_indices[:-1]):
        day_matrix = poi_cbg_visits_list[d:(d+24)]
        daily_sum = sum(day_matrix).sum(1)
        test = (daily_sum[daily_sum<5]>0).sum()
        daily_tests.append(test)
    return np.mean(daily_tests)


def population_file_test(region, filter_to_use):
    MIN_DATETIME,MAX_DATETIME = get_dates_by_phase(1)
    fn = f"{MIN_DATETIME.strftime('%Y-%m-%d')}_to_{MAX_DATETIME.strftime('%Y-%m-%d')}"
    STRATIFIED_BY_AREA_DIR = os.path.join(
        BASE_DIR, 
        'all_aggregate_data',
        'chunks_with_demographic_annotations_stratified_by_area', filter_to_use, fn)
    phase1_path = os.path.join(STRATIFIED_BY_AREA_DIR,f"{region}.csv")
    if os.path.exists(phase1_path):
        pop_phase1 = pd.read_csv(phase1_path)
    else:
        print("Phase 1 path does not exist. Exiting.")
        return
    
    MIN_DATETIME,MAX_DATETIME = get_dates_by_phase(2)
    fn = f"{MIN_DATETIME.strftime('%Y-%m-%d')}_to_{MAX_DATETIME.strftime('%Y-%m-%d')}"
    STRATIFIED_BY_AREA_DIR = os.path.join(
        BASE_DIR, 
        'all_aggregate_data',
        'chunks_with_demographic_annotations_stratified_by_area', filter_to_use, fn)
    phase2_path = os.path.join(STRATIFIED_BY_AREA_DIR,f"{region}.csv")
    if os.path.exists(phase2_path):
        pop_phase2 = pd.read_csv(phase2_path)
    else:
        print("Phase 2 path does not exist. Exiting.")    
        return
    return pop_phase1.equals(pop_phase2)
        