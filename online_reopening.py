from covid_constants_and_util import *
from disease_model import * 
import helper_methods_for_aggregate_data_analysis as helper
from model_experiments import *
from model_results import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
import pickle
from model_evaluation import evaluate_all_fitted_models_for_experiment, evaluate_all_fitted_models_for_msa
from run_one_model import get_full_activity_num_visits, apply_shift_in_days, apply_distancing_degree
from run_parallel_models import get_list_of_poi_subcategories_with_most_visits
from mobility_processing import get_metadata_filename
from search_model_results import get_experiments_results
import textwrap as tw

def get_combined_df_reopening_effects(
    intervention_df, msa_names, 
    poi_and_cbg_characteristics, cats_to_plot, alpha_target=0):
    """
    Make boxplots of the effects (on fraction infected) of opening each category of POI. 
    """
    assert len(msa_names) > 0
    print("Making plots using", msa_names)
    
    subcategory_counts = {}

    # each row in poi_characteristics_df is one POI. 
    poi_characteristics_dfs = []
    for msa in msa_names:
        print(msa)
        poi_characteristics = poi_and_cbg_characteristics[msa].copy()
        if 'poi_cbg_visits_list' in poi_characteristics:
            poi_cbg_visits_list = poi_characteristics['poi_cbg_visits_list']
        else:
            fn = get_ipf_filename(msa, MIN_DATETIME, MAX_DATETIME, True, True)
            print(fn)
            f = open(fn, 'rb')
            poi_cbg_visits_list = pickle.load(f)
            f.close()
        poi_cbg_visits_list = [m.transpose() for m in poi_cbg_visits_list]

        weighted_visits_over_area, squared_visits_over_area = get_poi_densities(
            poi_characteristics, poi_cbg_visits_list)
        msa_df = pd.DataFrame({
            'sub_category':poi_characteristics['poi_categories'],
            'original_dwell_times':poi_characteristics['poi_dwell_times'],
            'dwell_time_correction_factors':poi_characteristics['poi_dwell_time_correction_factors'],
            'weighted_visits_over_area':weighted_visits_over_area,
            'squared_visits_over_area':squared_visits_over_area})
        
        msa_df['pretty_name'] = msa_df['sub_category'].map(
            lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
        
        subcategory_counts[msa] = Counter(msa_df['pretty_name'])
        poi_characteristics_dfs.append(msa_df)
    
    try:
        poi_characteristics_df = pd.concat(poi_characteristics_dfs)
        poi_characteristics_df = poi_characteristics_df.loc[
            poi_characteristics_df['sub_category'].map(lambda x:x in cats_to_plot)]
        assert poi_characteristics_df.empty == False
    except AssertionError:
        poi_characteristics_df = pd.concat(poi_characteristics_dfs)
        poi_characteristics_df = poi_characteristics_df.loc[
            poi_characteristics_df['sub_category'].map(lambda x:x in map(str,cats_to_plot))]
        assert poi_characteristics_df.empty == False

    poi_characteristics_df['density*dwell_time_factor'] = poi_characteristics_df[
        'dwell_time_correction_factors'] * poi_characteristics_df['weighted_visits_over_area']
    poi_characteristics_df['visits^2*dwell_time_factor/area'] = poi_characteristics_df[
        'dwell_time_correction_factors'] * poi_characteristics_df['squared_visits_over_area']

    # each row in intervention_df is one fitted model. 
    intervention_df = intervention_df.copy()
    intervention_df = intervention_df.loc[intervention_df['MSA_name'].map(lambda x:x in msa_names)]
    total_modeled_pops = {} # number of people  we model in each MSA
    for msa in msa_names:
        ts = intervention_df.loc[intervention_df['MSA_name'] == msa, 'timestring'].iloc[0]
        model, _, _, _, _ = load_model_and_data_from_timestring(
                ts,
                load_fast_results_only=False,
                load_full_model=True)
        total_modeled_pops[msa] = model.CBG_SIZES.sum()

    intervention_df['pretty_cat_names'] = intervention_df[
        'counterfactual_sub_category'].map(
            lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
    intervention_df = intervention_df.loc[intervention_df[
        'counterfactual_sub_category'].map(lambda x:x in cats_to_plot)]
    # CHECK HERE
    full_reopen_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == 0]
    full_close_df = intervention_df.loc[intervention_df['counterfactual_alpha'] == alpha_target]
    merge_cols = ['model_fit_rank_for_msa', 'pretty_cat_names', 'MSA_name']

    # make each random seed into its own row. 
    full_close_df = unpack_random_seeds(full_close_df,
        cols_of_interest=['final infected fraction'],
        cols_to_keep=merge_cols)
    full_reopen_df = unpack_random_seeds(full_reopen_df,
        cols_of_interest=['final infected fraction'],
        cols_to_keep=merge_cols)

    merge_cols = merge_cols + ['random_seed']
    combined_df = pd.merge(
        full_reopen_df[merge_cols + ['final infected fraction']],
        full_close_df[merge_cols + ['final infected fraction']],
        on=merge_cols,
        validate='one_to_one',
        how='inner',
        suffixes=['_reopen', '_closed'])
    assert len(combined_df) == len(full_reopen_df) == len(full_close_df)
    combined_df['reopening_impact'] = (
        -combined_df['final infected fraction_reopen'] +combined_df['final infected fraction_closed'])
    combined_df['total_additional_infections_from_reopening'] = combined_df['reopening_impact'] * combined_df['MSA_name'].map(
        lambda x:total_modeled_pops[x])
    # multiply by 10^5 to get incidence
    combined_df['reopening_impact'] = combined_df['reopening_impact'] * INCIDENCE_POP
    print("Reopening impact quantifies cases per %i" % INCIDENCE_POP)

    n_pois_in_cat = []
    for i in range(len(combined_df)):
        n_pois_in_cat.append(subcategory_counts[combined_df['MSA_name'].iloc[i]]
                             [combined_df['pretty_cat_names'].iloc[i]])
    if sum(n_pois_in_cat) == 0:
        n_pois_in_cat = []
        for i in range(len(combined_df)):
            n_pois_in_cat.append(subcategory_counts[combined_df['MSA_name'].iloc[i]]
                                 [str(combined_df['pretty_cat_names'].iloc[i])])
        
    combined_df['n_pois_in_cat'] = n_pois_in_cat
    combined_df['reopening_impact_per_poi'] = combined_df['reopening_impact'] / combined_df['n_pois_in_cat']
    return combined_df, poi_characteristics_df

def get_mean_impact_reopening_effects(combined_df, msa_names):
    if len(msa_names) == 1:
        print("Stats on mean additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index()))
        print("Lower CI, additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .quantile(LOWER_PERCENTILE / 100)  # pandas quantile needs 0 < q <= 1
                       .sort_values(by='reopening_impact')[::-1].reset_index()))
        print("Upper CI, additional cases from reopening")
        print((combined_df[['pretty_cat_names', 'reopening_impact', 'total_additional_infections_from_reopening']]
                       .groupby(['pretty_cat_names'])
                       .quantile(UPPER_PERCENTILE / 100)
                       .sort_values(by='reopening_impact')[::-1].reset_index()))
        mean_impact = (combined_df[['pretty_cat_names', 'reopening_impact']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index())

        mean_impact_per_poi = (combined_df[['pretty_cat_names', 'reopening_impact_per_poi']]
                       .groupby(['pretty_cat_names'])
                       .mean()
                       .sort_values(by='reopening_impact_per_poi')[::-1].reset_index())
    else:
        # Want to make sure to weight each MSA equally, so have to take means twice.
        mean_impact = (combined_df[['MSA_name', 'pretty_cat_names', 'reopening_impact']]
                       .groupby(['pretty_cat_names', 'MSA_name'])
                       .mean()
                       .groupby('pretty_cat_names')
                       .mean()
                       .sort_values(by='reopening_impact')[::-1].reset_index())

        mean_impact_per_poi = (combined_df[['MSA_name', 'pretty_cat_names', 'reopening_impact_per_poi']]
                       .groupby(['pretty_cat_names', 'MSA_name'])
                       .mean()
                       .groupby('pretty_cat_names')
                       .mean()
                       .sort_values(by='reopening_impact_per_poi')[::-1].reset_index())

    return mean_impact, mean_impact_per_poi

def get_correlation_between_attributes(poi_characteristics_df, mean_impact_per_poi):
    mean_poi_characteristics = poi_characteristics_df[[
        'pretty_name', 'original_dwell_times', 'weighted_visits_over_area',
        'density*dwell_time_factor', 'squared_visits_over_area', 
        'visits^2*dwell_time_factor/area']].groupby('pretty_name').mean().reset_index()
    compute_correlations = pd.merge(
        mean_impact_per_poi, mean_poi_characteristics, 
        left_on='pretty_cat_names', right_on='pretty_name', 
        validate='one_to_one', how='inner')
    assert len(compute_correlations) == len(mean_poi_characteristics)
    
    print("Pearson correlations between attributes")
    print(compute_correlations.corr(method='pearson')['reopening_impact_per_poi'])
    print("Spearman correlations between attributes")
    print(compute_correlations.corr(method='spearman')['reopening_impact_per_poi'])
    return compute_correlations

def plot_reopening_effect_boxplot(
    combined_df, mean_impact, titlestring, 
    filename=None, only_plot_reopening_impact=False):
    # actually make box plots. 
    outlier_size = 1
    num_positive = np.sum(combined_df['reopening_impact'] > 0)
    print('%d / %d (num categories * seeds * model params) had reopening impact greater than 0' % (
        num_positive, len(combined_df)))
    print(combined_df.head())
    
    if not only_plot_reopening_impact:
        fig, axes = plt.subplots(2, 2, figsize=[15, 9])
        fig.subplots_adjust(wspace=10, hspace=0.5)
        for i, poi_characteristic in enumerate(['original_dwell_times', 'weighted_visits_over_area']):
            ax = axes[0][i]
            sns.boxplot(y="pretty_name",
                    x=poi_characteristic,
                    data=poi_characteristics_df,
                    order=list(mean_impact['pretty_cat_names']),
                    ax=ax,
                    fliersize=outlier_size)
            ax.set_ylabel("")
            if poi_characteristic == 'poi_areas':
                ax.set_xlabel("Area (sq feet)")
            elif poi_characteristic == 'original_dwell_times':
                ax.set_xlabel("Dwell time (minutes)")
                ax.set_xlim([0, 200])
            elif poi_characteristic == 'weighted_visits_over_area':
                ax.set_xlabel("Average visits per hour / sq ft")
                ax.set_xlim([1e-4, 1e-2])
            ax.grid(alpha=0.5)

        ax = axes[1][0]
        sns.boxplot(y="pretty_cat_names",
                    x="reopening_impact_per_poi",
                    data=combined_df,
                    order=mean_impact['pretty_cat_names'],
                    ax=ax,
                    whis=0,
                    fliersize=outlier_size)
        #ax.set_xscale('log')
        ax.set_ylabel("")
        ax.set_xlabel("Additional infections (per 100k), compared to not reopening (per POI)")
        #ax.set_xlim([1e-2, 10])
        ax.grid(alpha=0.5)
        #ax.grid(alpha=.5)
        
        ax = axes[1][1]
        sns.boxplot(y="pretty_cat_names",
                    x="reopening_impact",
                    data=combined_df,
                    order=mean_impact['pretty_cat_names'],
                    ax=ax,
                    whis=0,
                    fliersize=outlier_size)
        ax.set_xlabel("Additional infections (per 100k), compared to not reopening")
        #ax.set_xscale('log')
        ax.set_ylabel("")
        #ax.set_xlim([10, 1e4])
        fig.suptitle(titlestring)
        plt.subplots_adjust(wspace=.6)
        ax.grid(alpha=0.5)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(9.5,7))
        sns.boxplot(y="pretty_cat_names",
                    x="reopening_impact",
                    data=combined_df,
                    order=mean_impact['pretty_cat_names'],
                    ax=ax,
                    whis=0,
                    fliersize=outlier_size)
        ax.set_xlabel("Additional infections (per 100k),\ncompared to not reopening", fontsize=18)
        #ax.set_xscale('log')
        ax.set_ylabel("")
        ax.tick_params(labelsize=16)
        #ax.set_xlim([10, 1e4])
        ax.set_title(titlestring, fontsize=20)
        plt.subplots_adjust(wspace=.6)
        ax.grid(alpha=0.5)
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')   
        return fig,ax

def get_poi_and_cbg_characteristics(HIGHLIGHT_MSA, MIN_DATETIME, MAX_DATETIME):
    # reloads poi_and_cbg_characteristics
    p = get_metadata_filename(HIGHLIGHT_MSA, MIN_DATETIME, MAX_DATETIME, True, True)
    assert os.path.exists(p)
    print("metadata file last modified: %s" % time.ctime(os.path.getmtime(p)))

    f = open(p, 'rb')
    poi_and_cbg_characteristics = pickle.load(f)
    f.close()

    print(list(poi_and_cbg_characteristics.keys())[0])
    for k in ['municipalities', 'POI_municipalities', 'POI_categories']:
        print(len(poi_and_cbg_characteristics[HIGHLIGHT_MSA][k]), k)

    muns = set(poi_and_cbg_characteristics[HIGHLIGHT_MSA]['POI_municipalities'])
    M = len(muns)
    poi_areas = np.ones(M)*1076.39#np.random.normal(1076.39, 250, M)
    poi_areas_dict = {m:p for m,p in zip(muns, poi_areas)}

    poi_and_cbg_characteristics[HIGHLIGHT_MSA]['poi_areas'] = [
        poi_areas_dict[m] for m in poi_and_cbg_characteristics[HIGHLIGHT_MSA]['POI_municipalities']]

    poi_and_cbg_characteristics[HIGHLIGHT_MSA]['poi_categories'] = poi_and_cbg_characteristics[HIGHLIGHT_MSA]['POI_categories']

    dwell_times = np.ones(M)*75 # np.random.exponential(75,M)
    dwell_times_dict = {m:p for m,p in zip(muns, dwell_times)}
    poi_and_cbg_characteristics[HIGHLIGHT_MSA]['poi_dwell_times'] = [
        dwell_times_dict[m] 
        for m in poi_and_cbg_characteristics[HIGHLIGHT_MSA]['POI_municipalities']]

    poi_dwell_time_correction_factors = (dwell_times / (dwell_times+60)) ** 2
    poi_dwell_time_correction_factors_dict = {m:p for m,p in zip(muns, dwell_times)}
    poi_and_cbg_characteristics[HIGHLIGHT_MSA]['poi_dwell_time_correction_factors'] = [
        poi_dwell_time_correction_factors_dict[m] 
        for m in poi_and_cbg_characteristics[HIGHLIGHT_MSA]['POI_municipalities']]
    return poi_and_cbg_characteristics

def get_intervention_df(
    region, experiment='test_interventions', phase=2, min_t='2022_09_21'):
    res = get_experiments_results(
        experiment = experiment,
        phase=phase,
        region=region,
        min_t=min_t)

    min_timestring = res[0]
    pt1 = "_".join(res[-1].split("_")[:4])
    pt2 = "_"+str(int(res[-1].split("_")[4])+2)
    max_timestring = pt1+pt2

    required_data_kwargs = {'MSA_name': region, 'nrows':None}
    intervention_df = evaluate_all_fitted_models_for_experiment(
        experiment, 
        min_timestring=min_timestring, 
        max_timestring=max_timestring,
        required_data_kwargs = required_data_kwargs
    )

    intervention_df['MSA_name'] = intervention_df['data_kwargs'].map(lambda x:x['MSA_name'])
    model_pars = [
        'alpha', 'extra_weeks_to_simulate', 'intervention_datetime', 
        'top_category', 'sub_category']
    for k in model_pars:
        intervention_df['counterfactual_%s' % k] = intervention_df[
            'counterfactual_poi_opening_experiment_kwargs'].map(lambda x:x[k])

    intervention_df['model_fit_rank_for_msa'] = intervention_df['model_kwargs'].map(
        lambda x:x['model_quality_dict']['model_fit_rank_for_msa'])
    intervention_df['counterfactual_baseline_model'] = intervention_df['model_kwargs'].map(
        lambda x:x['model_quality_dict']['model_timestring'])
    intervention_df['how_to_select_best_grid_search_models'] = intervention_df['model_kwargs'].map(
        lambda x:x['model_quality_dict']['how_to_select_best_grid_search_models'])

    return intervention_df

def get_intervention_df_cases(intervention_df):
    intervention_df_cases = intervention_df.loc[
        intervention_df['how_to_select_best_grid_search_models'] == 'daily_cases_rmse'].copy()
    intervention_df_deaths = intervention_df.loc[
        intervention_df['how_to_select_best_grid_search_models'] == 'daily_deaths_rmse'].copy()
    intervention_df_deaths = intervention_df_deaths[
        intervention_df_deaths['timestring'] >= '2020_07_22_20']  # necessary bc emma ran some experiments on the same day
    intervention_df_poisson = intervention_df.loc[
        intervention_df['how_to_select_best_grid_search_models'] == 'daily_cases_poisson'].copy()

    print('Found %d models for selecting with RMSE cases, %d for RMSE deaths, %d for Poisson cases' % 
           (len(intervention_df_cases), len(intervention_df_deaths), len(intervention_df_poisson)))
    print(intervention_df_cases.groupby('MSA_name').size())

    intervention_df_cases['rmse_ratio'] = intervention_df_cases['model_kwargs'].map(
        lambda x:x['model_quality_dict']['ratio_of_loss_dict_daily_cases_RMSE_to_that_of_best_fitting_model'])
    intervention_df_cases_10 = intervention_df_cases[intervention_df_cases['rmse_ratio'] <= 1.10]
    print('Found %d models for selecting with RMSE cases 10%% threshold' % len(intervention_df_cases_10))
    return intervention_df_cases

def get_best_cat_to_plot(
    intervention_df, msa_names, poi_and_cbg_characteristics, 
    cats_to_plot, perc_1, perc_2, thresh=0.05, 
    stat='mean',perc_test=False, ks_test=False):
    combined_df_1, poi_characteristics_df = get_combined_df_reopening_effects(
        intervention_df, msa_names, poi_and_cbg_characteristics, 
        cats_to_plot, alpha_target=perc_1)

    combined_df_2, poi_characteristics_df = get_combined_df_reopening_effects(
        intervention_df, msa_names, poi_and_cbg_characteristics, 
        cats_to_plot, alpha_target=perc_2)

    res = []
    for c in cats_to_plot:
        x = combined_df_1.loc[combined_df_1['pretty_cat_names']==c, 'reopening_impact'].values
        y = combined_df_2.loc[combined_df_2['pretty_cat_names']==c, 'reopening_impact'].values
        if stat == 'mean':
            res_c = {
                'cat': c,
                'p1_median': np.mean(x), 
                'p2_median': np.mean(y), 
                'ks_test': stats.kruskal(x, y).pvalue}
        elif stat == 'median':
            res_c = {
                'cat': c,
                'p1_median': np.median(x), 
                'p2_median': np.median(y), 
                'ks_test': stats.kruskal(x, y).pvalue}
            
        res.append(res_c)

    res_df = pd.DataFrame(res)
    res_df_best = res_df.copy()
    if perc_test:
        res_df_best = res_df_best[
            (res_df['p1_median'] < res_df_best['p2_median'])]        
    if ks_test:
        res_df_best = res_df_best[            
            (res_df['ks_test'] <= thresh)]    
    return res_df_best

def get_data_to_plot(
    intervention_df, msa_names, poi_and_cbg_characteristics, 
    cats_to_plot_best, perc_1, perc_2):
    combined_df_1, poi_characteristics_df = get_combined_df_reopening_effects(
        intervention_df, msa_names, poi_and_cbg_characteristics, 
        cats_to_plot_best, alpha_target=perc_1)

    combined_df_2, poi_characteristics_df = get_combined_df_reopening_effects(
        intervention_df, msa_names, poi_and_cbg_characteristics, 
        cats_to_plot_best, alpha_target=perc_2)

    combined_df_1 = combined_df_1[['pretty_cat_names','reopening_impact']]
    combined_df_1['alpha'] = perc_1
    combined_df_2 = combined_df_2[['pretty_cat_names','reopening_impact']]
    combined_df_2['alpha'] = perc_2

    data_to_plot = pd.concat([combined_df_1, combined_df_2])
    return data_to_plot

def change_retail(x):
    if x == 'Retail trade, except of motor vehicles and motorcycles':
        return "Retail"
    elif ((x == 'Residential care activities') or (x == 'Human health activities')):
        return "Health"
    else:
        return "No Retail"


def get_data_to_plot_full(
    HIGHLIGHT_MSA, cats_to_plot, perc_1, perc_2, pval_thresh, ateco_code_ref, 
    simplify_results=True, stat='median',
    filter_p=False, filter_ks=False, min_t='2022_09_21'):
    msa_names = [HIGHLIGHT_MSA]
    poi_and_cbg_characteristics = get_poi_and_cbg_characteristics(
        HIGHLIGHT_MSA, MIN_DATETIME, MAX_DATETIME)    
    intervention_df = get_intervention_df(region=HIGHLIGHT_MSA, min_t=min_t)
    
    if (not filter_p) & (not filter_ks):
        cats_to_plot_best = cats_to_plot
    else:
        res_df_best = get_best_cat_to_plot(
            intervention_df, msa_names, poi_and_cbg_characteristics, 
            cats_to_plot, 
            perc_1=perc_1, perc_2=perc_2, 
            thresh=pval_thresh, stat=stat, 
            perc_test=filter_p, ks_test=filter_ks)
        cats_to_plot_best = res_df_best['cat'].tolist()
    
    data_to_plot = get_data_to_plot(
        intervention_df, msa_names, poi_and_cbg_characteristics, 
        cats_to_plot_best, perc_1, perc_2)    
    # formatting
    data_to_plot['pretty_cat_names'] = data_to_plot['pretty_cat_names'].apply(
        lambda x: ateco_code_ref.get(x))
    data_to_plot['reduction'] = data_to_plot['alpha'].apply(
        lambda x: f"{int((1+x)*100)}%")    
    if simplify_results:
        data_to_plot['pretty_cat_names'] = data_to_plot['pretty_cat_names'].apply(
            change_retail,1)
        new_order_s = ['Retail', 'No Retail', 'Health']
    else:
        combined_df, poi_characteristics_df = get_combined_df_reopening_effects(
            intervention_df, msa_names, poi_and_cbg_characteristics, 
            cats_to_plot_best, alpha_target=perc_2)
        if stat == 'median':
            new_order = combined_df.groupby(
                ['pretty_cat_names'])['reopening_impact'].median().sort_values(
                ascending=False).index.astype(str).tolist()
        elif stat == 'mean':
            new_order = combined_df.groupby(
                ['pretty_cat_names'])['reopening_impact'].mean().sort_values(
                ascending=False).index.astype(str).tolist()
        new_order_s = [ateco_code_ref[int(x)] for x in new_order]
    return data_to_plot, new_order_s

    #fig_lazio,ax_lazio = plt.subplots(1,1,figsize=(12,10))
    

    