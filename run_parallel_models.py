from covid_constants_and_util import *
from utilities import *
import helper_methods_for_aggregate_data_analysis as helper
import copy
import pickle
import re
import getpass
import psutil
import json
import subprocess
import multiprocessing
from psutil._common import bytes2human
import getpass
import time 
import random
from model_evaluation import *
from run_one_model import get_uniform_proportions_per_msa
import itertools

###################################################
# Code for running many models in parallel
###################################################
def generate_data_and_model_configs(
    msas_to_fit=10,
    config_idx_to_start_at=None,
    skip_previously_fitted_kwargs=False,
    min_timestring=None,
    min_timestring_to_load_best_fit_models_from_grid_search=None,
    max_timestring_to_load_best_fit_models_from_grid_search=None,
    experiment_to_run='normal_grid_search',
    how_to_select_best_grid_search_models=None,
    max_models_to_take_per_msa=MAX_MODELS_TO_TAKE_PER_MSA,
    acceptable_loss_tolerance=ACCEPTABLE_LOSS_TOLERANCE, 
    provided_msa=None):
    """
    Generates the set of parameter configurations for a given experiment.
    MSAs to fit: how many MSAs we will focus on.
    config_idx_to_start_at: how many configs we should skip.
    """
    # this controls what parameters we search over.
    config_generation_start_time = time.time()

    if skip_previously_fitted_kwargs:
        assert min_timestring is not None
        previously_fitted_timestrings = filter_timestrings_for_properties(min_timestring=min_timestring)
        previously_fitted_data_and_model_kwargs = [
            pickle.load(
                open(os.path.join(
                    FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb'))
            for timestring in previously_fitted_timestrings]
        print("Filtering out %i previously generated configs" % len(previously_fitted_data_and_model_kwargs))
    else:
        previously_fitted_data_and_model_kwargs = []

    # Helper dataframe to check current status of data
    # d = helper.load_chunk(1, load_backup=False)
    # d = helper.load_chunk(0, load_backup=False) #NEW

    # ---------------- Data kwargs ------------------
    # Store MSAs to load and how many rows to load
    data_kwargs = []

    # Load on largest N MSAs.
    # NEW_v2
    # biggest_msas = d['poi_lat_lon_CBSA Title'].value_counts().head(n=msas_to_fit)
    
    # biggest_msas = [
    #     'Milano', 'Varese', 'Torino', 'Roma', 'Napoli',
    #     'Monza e della Brianza', 'Bergamo', 'Padova', 
    #     'Treviso', 'Firenze']
    
    #biggest_msas = ['Milano','Roma','Napoli']
    if provided_msa is None:
        biggest_msas = BIGGEST_MSAS
    else:
        biggest_msas = [provided_msa]
        BIGGEST_MSAS = [provided_msa]

    print("Largest %i MSAs are" % len(biggest_msas))
    print(biggest_msas)
    
    # NEW_v2
    #for msa_name in biggest_msas.index:
    for msa_name in biggest_msas:
        name_without_spaces = re.sub('[^0-9a-zA-Z]+', '_', msa_name)
        data_kwargs.append({'MSA_name':name_without_spaces, 'nrows':None})
    
    # ---------------- Model kwargs ------------------       
    # NEW_v2
    # date_cols = [helper.load_date_col_as_date(a) for a in d.columns]
    # date_cols = [a for a in date_cols if a is not None]
    # max_date = max(date_cols)
    # max_datetime = datetime.datetime(max_date.year, max_date.month, max_date.day, 23)  # latest hour
    
    # CHECK: latest hour is manually fixed FTM   
    #min_datetime = datetime.datetime(2020, 2, 24, 0)  # min_datetime is fixed
    min_datetime = MIN_DATETIME
    max_datetime = MAX_DATETIME
    #max_datetime = datetime.datetime(2020, 5, 2, 23)
    print('Min datetime: %s. Max datetime: %s.' % (min_datetime, max_datetime))

    # Generate model kwargs. How exactly we do this depends on which experiments we're running.
    num_seeds = 90 # NEW_V4 for NHB
    #num_seeds = 30 # NEW GB added 20/05/23 for rebuttal
    configs_with_changing_params = []
    if experiment_to_run == 'just_save_ipf_output':
        model_kwargs = [
            {
                'min_datetime':min_datetime,
                'max_datetime':max_datetime,
                'exogenous_model_kwargs': {  
                    # could be anything, will not affect IPF
                    'home_beta':1e-2,
                    'poi_psi':1000,
                    'p_sick_at_t0':1e-4,
                    'just_compute_r0':False
                },
                'simulation_kwargs': {'do_ipf':True},
                'poi_attributes_to_clip':{                
                    'clip_areas':True,
                    'clip_dwell_times':True,
                    'clip_visits':True
                },
                'model_init_kwargs':{
                    'ipf_final_match':'poi',
                    'ipf_num_iter':100,
                    'num_seeds':num_seeds,
                },
                'include_cbg_prop_out':True
            }]

    elif experiment_to_run == 'normal_grid_search':
        # Run grid search on every combination of p0, beta, pis
        # TODO: change to account wave 2 situation where contagion is already spreading
        
        if DETECTION_LAG > 0:
            p_sicks = [ 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]        
        elif DETECTION_LAG == 0:
            p_sicks = [ 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6]        
        
        home_betas = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
        poi_psis = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], 
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)
        
        #mean_dwell_times = [65,70,75,80,85] # NEW_V4:  added GB 20220901
        mean_dwell_times = [75]

        # NEW_V4:  added GB 20220901
        #mus = range(750,1500,250)
        #sigmas = range(160,80*5,80)
        mus = [1250]
        sigmas = [320]
        update_betas = [6,10,14]
        
        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    for mean_dwell in mean_dwell_times:                        
                        for mu in  mus:
                            for sigma in sigmas:
                                for update_beta_b in update_betas:
                                    configs_with_changing_params.append({
                                        'home_beta':home_beta, 
                                        'poi_psi':poi_psi, 
                                        'p_sick_at_t0':p_sick,
                                        'mean_dwell':mean_dwell,
                                        'square_feet_mean': mu, # added GB 20220901
                                        'square_feet_sigma': sigma, # added GB 20220901
                                        'risk_update_param': update_beta_b, # added GB NEW_rebuttal_20240111 
                                    })
                        
        # ablation analyses.
        for home_beta in np.linspace(0.005, 0.04, 20):
            for p_sick in p_sicks:
                configs_with_changing_params.append({
                    'home_beta':home_beta, 
                    'poi_psi':0, 
                    'p_sick_at_t0':p_sick, 
                    'mean_dwell':69,
                    'square_feet_mean': 1076.39, # added GB 20220901
                    'square_feet_sigma': 240, # added GB 20220901
                    'risk_update_param': 14, # added GB NEW_rebuttal_20240111 
                })

    # NEW 20/05/2023
    # changed normal grid search for paper rebuttal 
    # this is a robust grid search
    # elif experiment_to_run == 'normal_grid_search': 
    #     # Run grid search on every combination of p0, beta, pis
    #     # TODO: change to account wave 2 situation where contagion is already spreading
        
    #     if DETECTION_LAG > 0:
    #         p_sicks = [ 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]        
    #     elif DETECTION_LAG == 0:
    #         p_sicks = [ 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6]        
        
    #     home_betas = np.linspace(
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
    #     poi_psis = np.linspace(
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], 
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)

    #     # NEW: added GB 20/05/23
    #     # restrict parameters space for rebuttal
    #     home_betas = home_betas[2:-2]
    #     poi_psis = poi_psis[1:]
    #     p_sicks = p_sicks[2:]        
        
    #     #mean_dwell_times = [65,70,75,80,85] # NEW_V4:  added GB 20220901
    #     # mean_dwell_times = [75]

    #     # NEW_V4:  added GB 20220901
    #     #mus = range(750,1500,250)
    #     #sigmas = range(160,80*5,80)

    #     # NEW: added GB 20/05/23
    #     dwell_model_dict = {
    #         'exponential': [65,75,85], #range(55,80,5),
    #         'lognormal': [30,25,35],#np.arange(25,37.5,2.5)
    #     }
    #     area_model_dict = {
    #         'constant': [861,1076,1291],
    #         'normal': [861,1076,1291],
    #         'uniform': [1076]
    #     }
    #     # mus = [1250]
    #     sigmas = [320]

    #     for home_beta in home_betas:
    #         for poi_psi in poi_psis:
    #             for p_sick in p_sicks:
    #                 for dwell_model, mean_dwell_times in dwell_model_dict.items():                                      
    #                     for mean_dwell in mean_dwell_times:    
    #                         for area_model, mus in area_model_dict.items():                    
    #                             for mu in  mus:
    #                                 for sigma in sigmas:                                        
    #                                     configs_with_changing_params.append({                                            
    #                                         'home_beta':home_beta, 
    #                                         'poi_psi':poi_psi, 
    #                                         'p_sick_at_t0':p_sick,
    #                                         'mean_dwell':mean_dwell,
    #                                         'square_feet_mean': mu, # added GB 20220901
    #                                         'square_feet_sigma': sigma, # added GB 20220901
    #                                         'dwell_model': dwell_model,
    #                                         'area_model': area_model    
    #                                         })
                        
    #     # ablation analyses.
    #     for home_beta in np.linspace(0.005, 0.04, 20):
    #         for p_sick in p_sicks:
    #             configs_with_changing_params.append({
    #                 'home_beta':home_beta, 
    #                 'poi_psi':0, 
    #                 'p_sick_at_t0':p_sick, 
    #                 'mean_dwell':69,
    #                 'square_feet_mean': 1076.39, # added GB 20220901
    #                 'square_feet_sigma': 240, # added GB 20220901
    #                 'dwell_model': dwell_model,
    #                 'area_model': area_model})
    
    # NEW 20/05/23
    # changed normal grid search for paper rebuttal 
    # elif experiment_to_run == 'normal_grid_search': 
    #     # Run grid search on every combination of p0, beta, pis
    #     # TODO: change to account wave 2 situation where contagion is already spreading
        
    #     if DETECTION_LAG > 0:
    #         p_sicks = [ 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]        
    #     elif DETECTION_LAG == 0:
    #         p_sicks = [ 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6]        
        
    #     home_betas = np.linspace(
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
    #     poi_psis = np.linspace(
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], 
    #         BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)

    #     # NEW: added GB 20/05/23
    #     # restrict parameters space for rebuttal
    #     home_betas = home_betas[2:-2]
    #     poi_psis = [ 1452.04836612, 1764.26365969, 2076.47895325, 2388.69424681]
    #     p_sicks = [0.002, 0.005]
    
    # #     home_betas = home_betas[2:-2]
    # #     poi_psis = poi_psis[1:]
    # #     p_sicks = p_sicks[2:]        

        
    #     #mean_dwell_times = [65,70,75,80,85] # NEW_V4:  added GB 20220901
    #     # mean_dwell_times = [75]

    #     # NEW_V4:  added GB 20220901
    #     #mus = range(750,1500,250)
    #     #sigmas = range(160,80*5,80)

    #     # NEW: added GB 20/05/23
    #     dwell_model_dict = {
    #         # 'exponential': [65,75,85], #range(55,80,5),
    #         'lognormal': [25,30,35], #[30,25,35],#np.arange(25,37.5,2.5)
    #     }
    #     area_model_dict = {
    #         'constant': [861,1076,1291],
    #         'normal': [861,1076,1291],
    #         # 'uniform': [1076]
    #     }
    #     # mus = [1250]
    #     sigmas = [320]

    #     for home_beta in home_betas:
    #         for poi_psi in poi_psis:
    #             for p_sick in p_sicks:
    #                 for dwell_model, mean_dwell_times in dwell_model_dict.items():                                      
    #                     for mean_dwell in mean_dwell_times:    
    #                         for area_model, mus in area_model_dict.items():                    
    #                             for mu in  mus:
    #                                 for sigma in sigmas:                                        
    #                                     configs_with_changing_params.append({                                            
    #                                         'home_beta':home_beta, 
    #                                         'poi_psi':poi_psi, 
    #                                         'p_sick_at_t0':p_sick,
    #                                         'mean_dwell':mean_dwell,
    #                                         'square_feet_mean': mu, # added GB 20220901
    #                                         'square_feet_sigma': sigma, # added GB 20220901
    #                                         'dwell_model': dwell_model,
    #                                         'area_model': area_model    
    #                                         })
                        
    #     # ablation analyses.
    #     for home_beta in np.linspace(0.005, 0.04, 20):
    #         for p_sick in p_sicks:
    #             configs_with_changing_params.append({
    #                 'home_beta':home_beta, 
    #                 'poi_psi':0, 
    #                 'p_sick_at_t0':p_sick, 
    #                 'mean_dwell':69,
    #                 'square_feet_mean': 1076.39, # added GB 20220901
    #                 'square_feet_sigma': 240, # added GB 20220901
    #                 'dwell_model': dwell_model,
    #                 'area_model': area_model})    

    elif experiment_to_run == 'grid_search_aggregate_mobility':
        p_sicks = [ 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]        
        home_betas = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
        poi_psis = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], 
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)
                    
        mean_dwell_times = [75]

        # NEW_V4:  added GB 20220901
        mus = range(500,1500,250)
        sigmas = range(80,80*5,80)

        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    for mean_dwell in mean_dwell_times:                        
                        for mu in  mus:
                            for sigma in sigmas:
                                configs_with_changing_params.append({
                                    'home_beta':home_beta, 
                                    'poi_psi':poi_psi, 
                                    'p_sick_at_t0':p_sick,
                                    'mean_dwell':mean_dwell,
                                    'square_feet_mean': mu, # added GB 20220901
                                    'square_feet_sigma': sigma # added GB 20220901
                                    })
            

    elif experiment_to_run == 'grid_search_home_proportion_beta':
        p_sicks = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
        home_betas = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_home_beta'],
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_home_beta'], 10)
        poi_psis = np.linspace(
            BETA_AND_PSI_PLAUSIBLE_RANGE['min_poi_psi'], 
            BETA_AND_PSI_PLAUSIBLE_RANGE['max_poi_psi'], 15)
        for home_beta in home_betas:
            for poi_psi in poi_psis:
                for p_sick in p_sicks:
                    configs_with_changing_params.append({
                        'home_beta':home_beta, 
                        'poi_psi':poi_psi, 
                        'p_sick_at_t0':p_sick})

    elif experiment_to_run == 'calibrate_r0':
        home_betas = [5e-2, 0.02, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 0]
        poi_psis = [
            20000, 15000, 10000,7500,6000, 5000,4500,4000,3500,
            3000,2500,2000,1500,1000,500,250,100]
        for home_beta in home_betas:
            configs_with_changing_params.append({
                'home_beta':home_beta, 
                'poi_psi':2500, 
                'p_sick_at_t0':1e-6, 
                'mean_dwell':75,
                'square_feet_mean': 1076.39, # added GB 20220901
                'square_feet_sigma': 240, # added GB 20220901
            })                
        for poi_psi in poi_psis:
            configs_with_changing_params.append({
                'home_beta':0.001, 
                'poi_psi':poi_psi, 
                'p_sick_at_t0':1e-6, 
                'mean_dwell':75,
                'square_feet_mean': 1076.39, # added GB 20220901
                'square_feet_sigma': 240 # added GB 20220901
            })

    elif experiment_to_run == 'calibrate_r0_aggregate_mobility':
        # home beta range will be the same as normal experiment
        poi_psis = [50, 25, 10, 5,  1, 0.5, 0.1, 0.005, 0.001]
        for poi_psi in poi_psis:
            configs_with_changing_params.append({
                'home_beta':0.001, 
                'poi_psi':poi_psi, 
                'p_sick_at_t0':1e-4})

    # experiments that require the best fit models
    best_models_experiments = {
        'test_interventions',
        'test_retrospective_counterfactuals',
        'test_retrospective_counterfactuals_rebuttal',
        'test_max_capacity_clipping',
        'test_uniform_proportion_of_full_reopening',
        'rerun_best_models_and_save_cases_per_poi',
        'test_categories_on_grid'
    }
    
    # ---------------- Merge data and model kwargs ------------------    
    # put this as another elif branch? 
    # No because we are merging model kwargs and data kwargs
    if experiment_to_run in best_models_experiments:
        # here model and data kwargs are entwined, so we can't just take 
        # the outer product of model_kwargs and data_kwargs.
        # this is because we load the best fitting model for each MSA.
        list_of_data_and_model_kwargs = []
        poi_categories_to_examine = 20
        if how_to_select_best_grid_search_models == 'daily_cases_rmse':
            key_to_sort_by = 'loss_dict_daily_cases_RMSE'
        elif how_to_select_best_grid_search_models == 'daily_deaths_rmse':
            key_to_sort_by = 'loss_dict_daily_deaths_RMSE'
        elif how_to_select_best_grid_search_models == 'daily_cases_poisson':
            key_to_sort_by = 'loss_dict_daily_cases_poisson_NLL_thres-10_sum'
        else:
            raise Exception("Not a valid means of selecting best-fit models")
        print("Selecting best grid search models using criterion %s" % how_to_select_best_grid_search_models)

        # get list of all fitted models:
        # need this for any of the "best fit models" experiments
        model_timestrings, model_msas = filter_timestrings_for_properties(
            min_timestring=min_timestring_to_load_best_fit_models_from_grid_search,
            max_timestring=max_timestring_to_load_best_fit_models_from_grid_search,
            required_properties={'experiment_to_run':'normal_grid_search'},
            return_msa_names=True)
        print("Found %i models after %s" % (
            len(model_timestrings), 
            min_timestring_to_load_best_fit_models_from_grid_search))
        timestring_msa_df = pd.DataFrame({
            'model_timestring':model_timestrings, 
            'model_msa':model_msas})
        n_models_for_msa_prior_to_quality_filter = None

        # get experiment-specific stuff
        if experiment_to_run == 'test_interventions':
            # most_visited_poi_subcategories = get_list_of_poi_subcategories_with_most_visits(
            #     n_poi_categories=poi_categories_to_examine)

            # New
            # use an hardcoded list: those contributing the most and those with 
            # greatest variation in online spending (24 categories)
            # most_visited_poi_subcategories = [
            #     100, 130, 230, 320, 330, 350, 450, 470, 520, 530,
            #     560, 590, 610, 640, 710, 750, 860, 870, 930, 960
            # ]
            
            # GB 230922: use a subset of sectors, eventually grouped together
            most_visited_poi_subcategories = [
                'retail',           # Commercio al dettaglio, escluso quello di autoveicoli e di motocicli (470)
                'transport',        # Trasporto: terrestre e mediante condotte (490) + marittimi e per vie d'acqua (500) + aereo (510)
                'accommodation',    # Servizi di alloggio (550)
                'restaurant',       # Attivita di servizi di ristorazione (560)
                'welfare',          # Attivita dei servizi sanitari (860) + Servizi di assistenza residenziale (870) + non residenziale (880) 
                'no_retail',        # No retail
                'other_sectors',    # All the rest
                'no_retail_no_welfare'
            ]

        else:
            most_visited_poi_subcategories = None
            
        if experiment_to_run == 'test_uniform_proportion_of_full_reopening':
            # need to match visits lost from max capacity clipping experiments
            msa2proportions = get_uniform_proportions_per_msa(
                min_timestring=min_timestring_to_load_best_fit_models_from_grid_search)
        else:
            msa2proportions = None

        # Loop over MSAs
        for row in data_kwargs:
            msa_t0 = time.time()
            msa_name = row['MSA_name']
            ### TODO: PROBLEM
            # timestrings=
            timestrings_for_msa = list(
                timestring_msa_df.loc[timestring_msa_df['model_msa'] == msa_name, 'model_timestring'].values)
            
            print("Evaluating %i timestrings for %s" % (len(timestrings_for_msa), msa_name))
            best_msa_models = evaluate_all_fitted_models_for_msa(
                msa_name, timestrings=timestrings_for_msa)

            best_msa_models = best_msa_models.loc[
                (best_msa_models['experiment_to_run'] == 'normal_grid_search') &
                (best_msa_models['poi_psi'] > 0)].sort_values(by=key_to_sort_by)

            if n_models_for_msa_prior_to_quality_filter is None:
                # make sure nothing weird happening / no duplicate models.
                n_models_for_msa_prior_to_quality_filter = len(best_msa_models) 
            else:
                assert len(best_msa_models) == n_models_for_msa_prior_to_quality_filter

            best_loss = float(best_msa_models.iloc[0][key_to_sort_by])
            print("After filtering for normal_grid_search models, %i models for MSA" % (len(best_msa_models)))
            best_msa_models = best_msa_models.loc[
                best_msa_models[key_to_sort_by] <= acceptable_loss_tolerance * best_loss]

            best_msa_models = best_msa_models.iloc[:max_models_to_take_per_msa]
            print("After filtering for models with %s within factor %2.3f of best loss, and taking max %i models, %i models" %
                (key_to_sort_by, 
                acceptable_loss_tolerance, 
                max_models_to_take_per_msa, 
                len(best_msa_models)))

            for i in range(len(best_msa_models)):
                loss_ratio = best_msa_models.iloc[i][key_to_sort_by]/best_loss
                assert loss_ratio >= 1 and loss_ratio <= acceptable_loss_tolerance
                model_quality_dict = {
                    'model_fit_rank_for_msa':i,
                    'how_to_select_best_grid_search_models':how_to_select_best_grid_search_models,
                    'ratio_of_%s_to_that_of_best_fitting_model' % key_to_sort_by:loss_ratio,
                    'model_timestring':best_msa_models.iloc[i]['timestring']}
                
                _, kwargs_i, _, _, _ = load_model_and_data_from_timestring(
                    best_msa_models.iloc[i]['timestring'], 
                    load_fast_results_only=True)
                
                kwargs_i['experiment_to_run'] = experiment_to_run
                del kwargs_i['model_kwargs']['counties_to_track']

                if experiment_to_run == 'test_retrospective_counterfactuals':
                    # LOOKING AT THE PAST.
                    # what if we had only done x% of social distancing?
                    # degree represents what percentage of social distancing to keep 
                    # (we don't need to test 100% because that is what actually happened)
                    for degree in [0, 0.25, 0.5, -1]: 
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'distancing_degree':degree}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                    # what if we shifted the timeseries by x days?
                    for shift in [-7, -3, 3, 7]:  # how many days to shift
                    #for shift in [-56, -28, -14, -7]:  # how many days to shift
                    #for shift in [-21, -14, 3, 7]:  # how many days to shift
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'shift_in_days':shift}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                    # sanity check
                    for percentage in [0.1,0.5,1,1.5,2.0]:  # how many days to shift
                    #for shift in [-56, -28, -14, -7]:  # how many days to shift
                    #for shift in [-21, -14, 3, 7]:  # how many days to shift
                        counterfactual_retrospective_experiment_kwargs = {'rescale_mobility':percentage}
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                # NEW_rebuttal_20240106
                if experiment_to_run == 'test_retrospective_counterfactuals_rebuttal':
                    # LOOKING AT THE PAST.
                    # what if we had only done x% of social distancing?
                    # degree represents what percentage of social distancing to keep 
                    # (we don't need to test 100% because that is what actually happened)
                    for degree in [0, 0.25, 0.5, -1, -2]: 
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'distancing_degree':degree}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                if experiment_to_run == 'test_categories_on_grid':
                    # LOOKING AT THE PAST.
                    # what if we reopen  up to the 2019 level?
                    # degree represents what percentage of social distancing to keep                     

                    target_cats = ['470', '560', '550', '860']
                    somelists = [list(itertools.product([tc],range(5))) for tc in target_cats]    
                    all_target_combs = list(itertools.product(*somelists))       

                    for cats in all_target_combs:       
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_retrospective_experiment_kwargs = {'grid_values': cats}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_retrospective_experiment_kwargs'] = counterfactual_retrospective_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)


                elif experiment_to_run == 'test_interventions':
                    # FUTURE EXPERIMENTS: reopen each category of POI.
                    for cat_idx in range(len(most_visited_poi_subcategories)):
                        # for alpha in [0, 1]:
                        #     kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        #     counterfactual_poi_opening_experiment_kwargs = {
                        #         'alpha':alpha,
                        #         'extra_weeks_to_simulate':4,
                        #         'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                        #         'top_category':None,
                        #         'sub_category':most_visited_poi_subcategories[cat_idx]}
                        #     kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()

                        #     kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        #     list_of_data_and_model_kwargs.append(kwarg_copy)

                        #for alpha in [1, .9, .75, .5]:
                        for alpha in [0, .1, .25, .5]:
                            kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                            counterfactual_poi_opening_experiment_kwargs = {
                                'alpha':alpha,
                                'extra_weeks_to_simulate':0, #NEW
                                'intervention_datetime':TRAIN_TEST_PARTITION, #NEW
                                'top_category':None,
                                'sub_category':most_visited_poi_subcategories[cat_idx]}
                            kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                            kwarg_copy['model_kwargs'][
                                'counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                            list_of_data_and_model_kwargs.append(kwarg_copy)


                elif experiment_to_run == 'test_max_capacity_clipping':
                    # FUTURE EXPERIMENTS: reopen fully but clip at alpha-proportion of max capacity
                    for alpha in np.arange(.1, 1.1, .1):
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                            'extra_weeks_to_simulate':4,
                            'intervention_datetime': TRAIN_TEST_PARTITION, #datetime.datetime(2020, 5, 1, 0),
                            'alpha':1,  # assume full activity before clipping
                            'max_capacity_alpha':alpha}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
                    # FUTURE EXPERIMENTS: test uniform reopening on all pois, simple proportion of pre-lockdown activity
                    for alpha in msa2proportions[msa_name]:
                        kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                        counterfactual_poi_opening_experiment_kwargs = {
                            'extra_weeks_to_simulate':4,
                            'intervention_datetime':datetime.datetime(2020, 5, 1, 0),
                            'full_activity_alpha':alpha}
                        kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                        kwarg_copy['model_kwargs']['counterfactual_poi_opening_experiment_kwargs'] = counterfactual_poi_opening_experiment_kwargs
                        list_of_data_and_model_kwargs.append(kwarg_copy)

                elif experiment_to_run == 'rerun_best_models_and_save_cases_per_poi':
                    # Rerun best fit models so that we can track the infection contribution of each POI,
                    # overall and for each income decile.
                    kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
                    simulation_kwargs = {
                        'groups_to_track_num_cases_per_poi':['all'],
                            #'median_household_income_bottom_decile',
                            #'median_household_income_top_decile']
                    }
                    kwarg_copy['model_kwargs']['simulation_kwargs'] = simulation_kwargs
                    kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()
                    list_of_data_and_model_kwargs.append(kwarg_copy)

            print("In total, it took %2.3f seconds to generate configs for MSA" % (time.time() - msa_t0))

        # sanity check to make sure nothing strange - number of parameters we expect.
        expt_params = []
        for row in list_of_data_and_model_kwargs:
            expt_params.append(
                {'home_beta':row['model_kwargs']['exogenous_model_kwargs']['home_beta'],
                 'poi_psi':row['model_kwargs']['exogenous_model_kwargs']['poi_psi'],
                 'p_sick_at_t0':row['model_kwargs']['exogenous_model_kwargs']['p_sick_at_t0'],
                 'MSA_name':row['data_kwargs']['MSA_name']})
        expt_params = pd.DataFrame(expt_params)
        n_experiments_per_param_setting = expt_params.groupby([
            'home_beta','poi_psi','p_sick_at_t0','MSA_name']).size()
        if experiment_to_run == 'test_interventions':
            #assert (n_experiments_per_param_setting.values == poi_categories_to_examine * 2).all()            
            try:
                assert (n_experiments_per_param_setting.values == poi_categories_to_examine * 4).all()
            except Exception:
                print(f"Test on number of parameters failed: {n_experiments_per_param_setting.values}")            
        elif experiment_to_run == 'test_max_capacity_clipping':
            assert (n_experiments_per_param_setting.values == 10).all()
        elif experiment_to_run == 'test_uniform_proportion_of_full_reopening':
            assert (n_experiments_per_param_setting.values == 10).all()
        elif experiment_to_run == 'rerun_best_models_and_save_cases_per_poi':
            assert (n_experiments_per_param_setting.values == 1).all()
        elif experiment_to_run == 'test_categories_on_grid':
            try:
                assert (n_experiments_per_param_setting.values == 8).all()
            except Exception:
                print(f"Test on number of parameters failed: {n_experiments_per_param_setting.values}")
        elif experiment_to_run == 'test_retrospective_counterfactuals':
            #assert (n_experiments_per_param_setting.values == 8).all()
            assert (n_experiments_per_param_setting.values == 13).all() #NEW 05/05
        else:
            try:
                assert (n_experiments_per_param_setting.values == 8).all()
            except Exception:
                print(
                    f"Test on number of parameters failed "+
                    f"for experiment {experiment_to_run} " + 
                    f": {n_experiments_per_param_setting.values}")

    else:  # if experiment_to_run is not in best_models_experiments
        if experiment_to_run != 'just_save_ipf_output':  # model_kwargs is already set for ipf experiment
            model_kwargs = []
            for config in configs_with_changing_params:
                # NEW: 20/05/23
                if 'area_model' not in config:
                    config['area_model'] = 'normal'
                if 'dwell_model' not in config:
                    config['dwell_model'] = 'exponential'

                model_kwargs.append({
                    'min_datetime':min_datetime,
                    'max_datetime':max_datetime,
                    'exogenous_model_kwargs': {
                        'home_beta':config['home_beta'],
                        'poi_psi':config['poi_psi'],
                        'p_sick_at_t0':config['p_sick_at_t0'],
                        'mean_dwell':config['mean_dwell'], 
                        'square_feet_mean': config['square_feet_mean'], # added GB 20220901
                        'square_feet_sigma': config['square_feet_sigma'], # added GB 20220901
                        'just_compute_r0':'calibrate_r0' in experiment_to_run,
                        'dwell_model': config['dwell_model'], # NEW: 20/05/23
                        'area_model': config['area_model'], # NEW: 20/05/23
                        'risk_update_param': config['risk_update_param'], # NEW: 11/01/24
                    },
                    'model_init_kwargs':{'num_seeds':num_seeds,},
                    'simulation_kwargs':{
                        'use_aggregate_mobility':'aggregate_mobility' in experiment_to_run,
                        'use_home_proportion_beta':'home_proportion_beta' in experiment_to_run,},
                    'poi_attributes_to_clip':{
                        'clip_areas':True,
                        'clip_dwell_times':True,
                        'clip_visits':True},
                    'include_cbg_prop_out':'home_proportion_beta' in experiment_to_run,
                })
        list_of_data_and_model_kwargs = [
            {
                'data_kwargs':copy.deepcopy(a), 
                'model_kwargs':copy.deepcopy(b), 
                'experiment_to_run':experiment_to_run
            } 
            for b in model_kwargs 
            for a in data_kwargs]

    # remove previously fitted kwargs
    if len(previously_fitted_data_and_model_kwargs) > 0:
        print("Prior to filtering out previously fitted kwargs, %i kwargs" % len(list_of_data_and_model_kwargs))
        for i in range(len(previously_fitted_data_and_model_kwargs)):
            # remove stuff that is added when saved so configs are comparable.
            if 'counties_to_track' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
                del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['counties_to_track']
            #if 'preload_poi_visits_list_filename' in previously_fitted_data_and_model_kwargs[i]['model_kwargs']:
            #    del previously_fitted_data_and_model_kwargs[i]['model_kwargs']['preload_poi_visits_list_filename']

        old_len = len(list_of_data_and_model_kwargs)
        list_of_data_and_model_kwargs = [
            a for a in list_of_data_and_model_kwargs if a not in previously_fitted_data_and_model_kwargs]
        assert old_len != len(list_of_data_and_model_kwargs)
        print("After removing previously fitted kwargs, %i kwargs" % (len(list_of_data_and_model_kwargs)))

    print("Total data/model configs to fit: %i; randomly shuffling order" % len(list_of_data_and_model_kwargs))
    random.Random(0).shuffle(list_of_data_and_model_kwargs)
    if config_idx_to_start_at is not None:
        print("Skipping first %i configs" % config_idx_to_start_at)
        list_of_data_and_model_kwargs = list_of_data_and_model_kwargs[config_idx_to_start_at:]
    print("Total time to generate configs: %2.3f seconds" % (time.time() - config_generation_start_time))
    return list_of_data_and_model_kwargs


def get_list_of_poi_subcategories_with_most_visits(n_poi_categories, n_chunks=5, return_df_without_filtering_or_sorting=False):
    """
    Return n_poi_categories subcategories with the most visits in "normal times" (Jan 2019 - Feb 2020)
    """
    normal_times = helper.list_datetimes_in_range(datetime.datetime(2019, 1, 1),
                                              datetime.datetime(2020, 2, 29))
    normal_time_cols = ['%i.%i.%i' % (a.year, a.month, a.day) for a in normal_times]
    must_have_cols = normal_time_cols + ['sub_category', 'top_category']
    d = helper.load_multiple_chunks(range(n_chunks), cols=must_have_cols)
    d['visits_in_normal_times'] = d[normal_time_cols].sum(axis=1)
    if return_df_without_filtering_or_sorting:
        d = d[['sub_category', 'visits_in_normal_times']]
        grouped_d = d.groupby(['sub_category']).agg(['sum', 'size']).reset_index()
        grouped_d.columns = ['Original Name', 'N visits', 'N POIs']
        grouped_d['% POIs'] = 100 * grouped_d['N POIs'] / grouped_d['N POIs'].sum()
        grouped_d['% visits'] = 100 * grouped_d['N visits'] / grouped_d['N visits'].sum()
        grouped_d['Category'] = grouped_d['Original Name'].map(lambda x:SUBCATEGORIES_TO_PRETTY_NAMES[x] if x in SUBCATEGORIES_TO_PRETTY_NAMES else x)
        grouped_d = grouped_d.sort_values(by='% visits')[::-1].head(n=n_poi_categories)[['Category', '% visits', '% POIs', 'N visits', 'N POIs']]
        print('Percent of POIs: %2.3f; percent of visits: %2.3f' %
              (grouped_d['% POIs'].sum(),
               grouped_d['% visits'].sum()))
        return grouped_d
    assert((d.groupby('sub_category')['top_category'].nunique().values == 1).all()) # Make sure that each sub_category only maps to one top category (and so it's safe to just look at sub categories).
    d = d.loc[d['sub_category'].map(lambda x:x not in REMOVED_SUBCATEGORIES)]
    grouped_d = d.groupby('sub_category')['visits_in_normal_times'].sum().sort_values()[::-1].iloc[:n_poi_categories]
    print("Returning the list of %i POI subcategories with the most visits, collectively accounting for percentage %2.1f%% of visits" %
        (n_poi_categories, 100*grouped_d.values.sum()/d['visits_in_normal_times'].sum()))
    return list(grouped_d.index)

def check_memory_usage():
    virtual_memory = psutil.virtual_memory()
    total_memory = getattr(virtual_memory, 'total')
    available_memory = getattr(virtual_memory, 'available')
    free_memory = getattr(virtual_memory, 'free')
    available_memory_percentage = 100. * available_memory / total_memory
    # Free memory is the amount of memory which is currently not used for anything. This number should be small, because memory which is not used is simply wasted.
    # Available memory is the amount of memory which is available for allocation to a new process or to existing processes.
    print('Total memory: %s; free memory: %s; available memory %s; available memory %2.3f%%' % (
        bytes2human(total_memory),
        bytes2human(free_memory),
        bytes2human(available_memory),
        available_memory_percentage))
    return available_memory_percentage

def run_many_models_in_parallel(configs_to_fit):
    # assign appropriate CPUs to the experiment
    max_processes_for_user = int(multiprocessing.cpu_count() / 1.2)
    print("Maximum number of processes to run: %i" % max_processes_for_user)

    # loop over config using their position as id
    for config_idx in range(len(configs_to_fit)):
        t0 = time.time()
        # Check how many processes user is running.        
        process_to_check = 'model_experiments.py'
        # Linux version
        shell_command = f'ps -fA | grep {process_to_check} | wc -l'
        # shell_command = f'wmic process where caption="python.exe" get commandline | find /i /c "{process_to_check}"'
        n_processes_running = int(subprocess.check_output(shell_command, shell=True))

        print("Current processes running for user: %i" % n_processes_running)
        while n_processes_running > max_processes_for_user:
            print("Current processes are %i, above threshold of %i; waiting." % (n_processes_running, max_processes_for_user))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            #n_processes_running = int(subprocess.check_output('ps -fA | grep model_experiments.py | wc -l', shell=True))
            n_processes_running = int(subprocess.check_output(shell_command, shell=True))

        # don't swamp cluster. Check CPU usage.
        cpu_usage = psutil.cpu_percent()
        print("Current CPU usage: %2.3f%%" % cpu_usage)
        while cpu_usage > CPU_USAGE_THRESHOLD:
            print("Current CPU usage is %2.3f, above threshold of %2.3f; waiting." % (cpu_usage, CPU_USAGE_THRESHOLD))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            cpu_usage = psutil.cpu_percent()

        # Also check memory.
        available_memory_percentage = check_memory_usage()
        while available_memory_percentage < 100 - MEM_USAGE_THRESHOLD:
            print("Current memory usage is above threshold of %2.3f; waiting." % (MEM_USAGE_THRESHOLD))
            time.sleep(SECONDS_TO_WAIT_AFTER_EXCEEDING_COMP_THRESHOLD)
            available_memory_percentage = check_memory_usage()

        # If we pass these checks, start a job.
        timestring = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
        experiment_to_run = configs_to_fit[config_idx]['experiment_to_run']
        print("Starting job %i/%i" % (config_idx + 1, len(configs_to_fit)))
        outfile_path = os.path.join(FITTED_MODEL_DIR, 'model_fitting_logfiles/%s.txt' % timestring)
        
        # Finally launch the job from command line
        cmd = 'nohup python3 -u model_experiments.py fit_and_save_one_model %s --timestring %s --config_idx %i > %s 2>&1 &' % (experiment_to_run, timestring, config_idx, outfile_path)
        # cmd = 'start /b python -u model_experiments.py fit_and_save_one_model %s --timestring %s --config_idx %i > %s 2>&1 &' % (
        #     experiment_to_run, timestring, config_idx, outfile_path)
        
        print("Command: %s" % cmd)
        os.system(cmd)
        time.sleep(3)
        print("Time between job submissions: %2.3f" % (time.time() - t0))

def print_config_as_json(data_and_model_config):
    data_and_model_config = copy.deepcopy(data_and_model_config)
    for k in data_and_model_config:
        if type(data_and_model_config[k]) is dict:
            for k1 in data_and_model_config[k]:
                data_and_model_config[k][k1] = str(data_and_model_config[k][k1])
        else:
            data_and_model_config[k] = str(data_and_model_config[k])
    print(json.dumps(data_and_model_config, indent=4, sort_keys=True))
    
def get_computer_and_resources_to_run(username, computer_name, default_core_number=2):    
    if username in COMPUTER_DICT:
        computers_to_use = COMPUTER_DICT[username]
        computer_stats = COMPUTER_STATS
        # computers_to_use = computer_dict[username]
        # # TODO: move this to config.
        # computer_stats = {
        #     'DESKTOP-2U7PSI7':4,
        #     'MBPdiFrancesco2.lan': 2,
        #     'CRISIS4':60,
        #     'rambo':288,
        #     'trinity':144,
        #     'furiosa':144,
        #     'madmax':64,
        #     'madmax2':80,
        #     'madmax3':80,
        #     'madmax4':80,
        #     'madmax5':80,
        #     'madmax6':80,
        #     'madmax7':80}
    else:
        #raise Exception("Please specify what computers you want to use!")
        print(f"Computer not found in the available configurations.")
        print(f"Assigned {default_core_number} cores by default ({multiprocessing.cpu_count()} available).")
        computers_to_use = [computer_name]
        computer_stats = {computer_name: default_core_number}        
    return computers_to_use, computer_stats

def partition_jobs_across_computers(computer_name, configs_to_fit):
    computer_name = computer_name.replace('.stanford.edu', '')
    username = getpass.getuser()
    print(f"Provided username: {username}")    
    computers_to_use, computer_stats = get_computer_and_resources_to_run(
        username, computer_name)
    total_cores = sum([computer_stats[a] for a in computers_to_use])
    computer_loads = dict([(k, computer_stats[k]/total_cores) for k in computers_to_use])
    print('Partitioning up jobs among computers as follows', computer_loads)
    assert computer_name in computer_loads
    assert np.allclose(sum(computer_loads.values()), 1)
    start_idx = 0
    computers_to_configs = {}
    for computer_idx, computer in enumerate(sorted(computer_loads.keys())):
        if computer_idx == len(computer_loads) - 1:
            computers_to_configs[computer] = configs_to_fit[start_idx:]
        else:
            end_idx = start_idx + int(len(configs_to_fit) * computer_loads[computer])
            computers_to_configs[computer] = configs_to_fit[start_idx:end_idx]
            start_idx = end_idx
    assert sum([len(a) for a in computers_to_configs.values()]) == len(configs_to_fit)
    print("Assigning %i configs to %s" % (len(computers_to_configs[computer_name]), computer_name))
    return computers_to_configs[computer_name]
