from covid_constants_and_util import *
from utilities import *
from disease_model import Model
import helper_methods_for_aggregate_data_analysis as helper
from collections import Counter
import pickle
import re
from scipy.stats import scoreatpercentile, lognorm
from scipy.sparse import hstack
from collections import Counter
import time 
import math 
import gzip

#from model_evaluation import *
from model_evaluation import get_variables_for_evaluating_msa_model, make_slir_race_ses_plot,compare_model_vs_real_num_cases, evaluate_all_fitted_models_for_experiment, load_model_and_data_from_timestring, sanity_check_error_metrics, get_datetimes_and_totals_from_nyt_outcomes
from mobility_processing import manage_exporting_paths, get_metadata_filename, get_ipf_filename
###################################################
# Code for running one model
###################################################

def fit_and_save_one_model(timestring,
                           model_kwargs,
                           data_kwargs,
                           d=None,
                           experiment_to_run=None,
                           train_test_partition=None,
                           filter_for_cbgs_in_msa=False):
    '''
    Fits one model, saves its results and evaluations of the results.  
    
    OVERSIMPLIFIED VERSION of the function in the previous version. Just give us the model output. We'll deal 
    with the rest later down the road.
    
    --- Input ---
    timestring : str
        To use in filenames to identify the model and its config. If None, then the model is not saved.
    model_kwargs : dict
        Arguments to use for fit_disease_model_on_real_data. 
        required keys: min_datetime, max_datetime, exogenous_model_kwargs, poi_attributes_to_clip.
    data_kwargs : dict
        Arguments for the data; required to have key 'MSA_name'.
    d : pandas DataFrame
        The dataframe for the MSA pois; if None, then the dataframe is loaded within the function.
    experiment_to_run:  str 
        Name of experiment to run.
    train_test_partition: DateTime object
        The first hour of test; if included, then losses are saved separately for train and test dates.
    filter_for_cbgs_in_msa : bool
        Whether to only model CBGs in the MSA
    '''
    #----- Argument checks -----
    must_have_args = [
        'min_datetime', 'max_datetime', 'exogenous_model_kwargs','poi_attributes_to_clip']
    assert all([k in model_kwargs for k in must_have_args])
    assert 'MSA_name' in data_kwargs    
    
    t0 = time.time()
    return_without_saving = False
    if timestring is None:
        print("Fitting single model. Timestring is none so not saving model and just returning fitted model.")
        return_without_saving = True
    else:
        print("Fitting single model. Results will be saved using timestring %s" % timestring)
    
    #----- Load data -----
    # TODO: understand what is needed from d and load it in other ways
    if d is None:  
        #d = helper.load_dataframe_for_individual_msa(**data_kwargs)
        d = helper.load_dataframe_for_individual_province(**data_kwargs)
    
    meta_d = helper.load_metadata_for_individual_province(**data_kwargs)


    # TODO: understand what is needed from here and load it in other ways
    # nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(
    #     data_kwargs['MSA_name'])
    nyt_outcomes = get_variables_for_evaluating_province_model(data_kwargs['MSA_name'])
    
    #----- Define experiment options -----
    # DO NOT track specific counties
    model_kwargs['counties_to_track'] = None
    # DO NOT filter for CBGs within MSA    
    cbgs_to_filter_for = None
    # ALWAYS correct visits 
    correct_visits = True
    # DO NOT track specific CBGs
    cbg_groups_to_track = {} 
    cbg_groups_to_track['nyt'] = None
    # ALWAYS default to attempting to load file.
    preload_poi_visits_list_filename = get_ipf_filename(
        msa_name=data_kwargs['MSA_name'],
        min_datetime=model_kwargs['min_datetime'],
        max_datetime=model_kwargs['max_datetime'],
        clip_visits=True,#model_kwargs['poi_attributes_to_clip']['clip_visits'], #TODO: change this back
        correct_visits=correct_visits)
    
    if not os.path.exists(preload_poi_visits_list_filename):
        print("Warning: path %s does not exist; exiting" % preload_poi_visits_list_filename)
        return
    else:
        print("Reloading POI visits from %s" % preload_poi_visits_list_filename)    
        model_kwargs['preload_poi_visits_list_filename'] = preload_poi_visits_list_filename

    #----- Fit model -----
    fitted_model = fit_disease_model_on_real_data(
        d,
        cbg_groups_to_track=cbg_groups_to_track,
        cbgs_to_filter_for=cbgs_to_filter_for,
        meta_d=meta_d,
        msa_name=data_kwargs['MSA_name'],
        **model_kwargs)
    
    #----- Manage model results -----

    #----- 1. NEVER Return IPF output (because we never run it) -----
    #----- 2. NEVER Just return the output -----
    #----- 3. NEVER Evaluate results broken down by race and SES.
    #     
    #----- 4. Save results--------    
    #----- 4a. ALWAYS Save model--------
    mdl_path = os.path.join(
        FITTED_MODEL_DIR, 'full_models', 
        'fitted_model_%s.gzip' % timestring) #'fitted_model_%s.pkl' % timestring
    print("Saving model at %s..." % mdl_path)

    with gzip.open(mdl_path, "wb") as file:
        #file = open(mdl_path, 'wb')
        fitted_model.save(file)
        #file.close()

    #----- 4b. TRY to save separate model results (see what fails here) --------
    model_results_to_save_separately = {}
    #for attr_to_save_separately in ['history', 'CBGS_TO_IDXS']:
    for attr_to_save_separately in ['history']:
        model_results_to_save_separately[attr_to_save_separately] = getattr(fitted_model, attr_to_save_separately)

    if SAVE_MODEL_RESULTS_SEPARATELY:
        # Save some smaller model results for quick(er) loading. 
        # For really fast stuff, like losses (numerical results only) we store separately.
        print("Saving model results...")
        file = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'wb')
        pickle.dump(model_results_to_save_separately, file)
        file.close()
    
    #----- 4c. ALWAYS Save separate descriptive results --------
    # evaluate model fit to cases and save loss separately as well.
    # Everything saved in this data structure should be a summary result - small and fast to load, numbers only!
    print("# ---- Estimating losses - 1. all sample")
    loss_dict,_ = compare_model_vs_real_num_cases(
        nyt_outcomes,
        model_kwargs['min_datetime'],
        model_results=model_results_to_save_separately)
    
    # fast_to_load_results initialized  here
    fast_to_load_results = {'loss_dict':loss_dict}
    
    if train_test_partition is not None:
        print("# ---- Estimating losses - 2 train set")
        train_max = train_test_partition + datetime.timedelta(hours=-1)
        print(train_test_partition)
        print(train_max)
        train_loss_dict,_ = compare_model_vs_real_num_cases(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            compare_end_time = train_max,
            model_results=model_results_to_save_separately)
        fast_to_load_results['train_loss_dict'] = train_loss_dict
        
        print("# ---- Estimating losses - 3 test set")
        test_loss_dict,_ = compare_model_vs_real_num_cases(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            compare_start_time = train_test_partition,
            model_results=model_results_to_save_separately)
        fast_to_load_results['test_loss_dict'] = test_loss_dict
        
        fast_to_load_results['train_test_date_cutoff'] = train_test_partition
        sanity_check_error_metrics(fast_to_load_results)

    fast_to_load_results['clipping_monitor'] = fitted_model.clipping_monitor
    final_infected_fraction = (
        fitted_model.cbg_infected + fitted_model.cbg_removed + fitted_model.cbg_latent).sum(axis=1)/fitted_model.CBG_SIZES.sum()
    fast_to_load_results['final infected fraction'] = final_infected_fraction    
    fast_to_load_results['estimated_R0'] = fitted_model.estimated_R0
    fast_to_load_results['intervention_cost'] = fitted_model.INTERVENTION_COST
    file = open(os.path.join(FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()

    #----- 4c. ALWAYS Save kwargs.--------
    data_and_model_kwargs = {'model_kwargs':model_kwargs, 'data_kwargs':data_kwargs, 'experiment_to_run':experiment_to_run}
    file = open(os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'wb')
    pickle.dump(data_and_model_kwargs, file)
    file.close()
    print("Successfully fitted and saved model and data_and_model_kwargs; total time taken %2.3f seconds" % (time.time() - t0))
    return fitted_model



def fit_and_save_one_model_OLD(timestring,
                           model_kwargs,
                           data_kwargs,
                           d=None,
                           experiment_to_run=None,
                           train_test_partition=None,
                           filter_for_cbgs_in_msa=False):
    '''
    Fits one model, saves its results and evaluations of the results.    
    
    --- Input ---
    timestring : str
        To use in filenames to identify the model and its config. If None, then the model is not saved.
    model_kwargs : dict
        Arguments to use for fit_disease_model_on_real_data. 
        required keys: min_datetime, max_datetime, exogenous_model_kwargs, poi_attributes_to_clip.
    data_kwargs : dict
        Arguments for the data; required to have key 'MSA_name'.
    d : pandas DataFrame
        The dataframe for the MSA pois; if None, then the dataframe is loaded within the function.
    experiment_to_run:  str 
        Name of experiment to run.
    train_test_partition: DateTime object
        The first hour of test; if included, then losses are saved separately for train and test dates.
    filter_for_cbgs_in_msa : bool
        Whether to only model CBGs in the MSA
    '''
    #----- Argument checks -----
    must_have_args = [
        'min_datetime', 'max_datetime', 'exogenous_model_kwargs','poi_attributes_to_clip']
    assert all([k in model_kwargs for k in must_have_args])
    assert 'MSA_name' in data_kwargs    
    
    t0 = time.time()
    return_without_saving = False
    if timestring is None:
        print("Fitting single model. Timestring is none so not saving model and just returning fitted model.")
        return_without_saving = True
    else:
        print("Fitting single model. Results will be saved using timestring %s" % timestring)
    
    #----- Load data -----
    if d is None:  
        d = helper.load_dataframe_for_individual_msa(**data_kwargs)
    
    nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(
        data_kwargs['MSA_name'])
    
    #----- Define experiment options -----
    # track specific counties
    if 'counties_to_track' not in model_kwargs:
        model_kwargs['counties_to_track'] = msa_counties
    
    # track specific CBGs
    cbg_groups_to_track = {}
    cbg_groups_to_track['nyt'] = nyt_cbgs
    
    if filter_for_cbgs_in_msa:
        print("Filtering for %i CBGs within MSA %s" % (len(msa_cbgs), data_kwargs['MSA_name']))
        cbgs_to_filter_for = set(msa_cbgs) # filter for CBGs within MSA
    else:
        cbgs_to_filter_for = None
    correct_visits = model_kwargs['correct_visits'] if 'correct_visits' in model_kwargs else True  # default to True
    
    # preload 
    if experiment_to_run == 'just_save_ipf_output':
        # If we're saving IPF output, don't try to reload file.
        preload_poi_visits_list_filename = None
    elif 'poi_cbg_visits_list' in model_kwargs:
        print('Passing in poi_cbg_visits_list, will not load from file')
        preload_poi_visits_list_filename = None 
    else:
        # Otherwise, default to attempting to load file.
        preload_poi_visits_list_filename = get_ipf_filename(
            msa_name=data_kwargs['MSA_name'],
            min_datetime=model_kwargs['min_datetime'],
            max_datetime=model_kwargs['max_datetime'],
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'],
            correct_visits=correct_visits)
        if not os.path.exists(preload_poi_visits_list_filename):
            print("Warning: path %s does not exist; regenerating POI visits" % preload_poi_visits_list_filename)
            preload_poi_visits_list_filename = None
        else:
            print("Reloading POI visits from %s" % preload_poi_visits_list_filename)
    model_kwargs['preload_poi_visits_list_filename'] = preload_poi_visits_list_filename

    #----- Fit model -----
    fitted_model = fit_disease_model_on_real_data(
        d,
        cbg_groups_to_track=cbg_groups_to_track,
        cbgs_to_filter_for=cbgs_to_filter_for,
        msa_name=data_kwargs['MSA_name'],
        **model_kwargs)
    
    #----- Manage model results -----

    #----- 1. Return IPF output -----
    if experiment_to_run == 'just_save_ipf_output':
        pickle_start_time = time.time()
        ipf_filename = get_ipf_filename(
            msa_name=data_kwargs['MSA_name'],
            min_datetime=model_kwargs['min_datetime'],
            max_datetime=model_kwargs['max_datetime'],
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'],
            correct_visits=correct_visits)
        
        print('Saving IPF output in', ipf_filename)
        ipf_file = open(ipf_filename, 'wb')
        pickle.dump(fitted_model.poi_cbg_visit_history, ipf_file)
        ipf_file.close()

        print('Time to save pickle = %.2fs' % (time.time() - pickle_start_time))
        print('Size of pickle: %.2f MB' % (os.path.getsize(ipf_filename) / (1024**2)))
        return
    #----- 2. Just return the output -----
    if return_without_saving:
        return fitted_model

    #----- 3. Evaluate results broken down by race and SES.
    plot_path = os.path.join(
        helper.FITTED_MODEL_DIR, 'ses_race_plots', 'ses_race_plot_%s.pdf' % timestring)
    ses_race_results = make_slir_race_ses_plot(fitted_model, path_to_save=plot_path)
    fitted_model.SES_RACE_RESULTS = ses_race_results
    
    #----- 4. Save results--------    
    #----- 4a. Save model--------
    mdl_path = os.path.join(
        FITTED_MODEL_DIR, 'full_models', 'fitted_model_%s.pkl' % timestring)
    print("Saving model at %s..." % mdl_path)
    file = open(mdl_path, 'wb')
    fitted_model.save(file)
    file.close()

    #----- 4b. Save separate model results --------
    model_results_to_save_separately = {}
    for attr_to_save_separately in ['history', 'CBGS_TO_IDXS']:
        model_results_to_save_separately[attr_to_save_separately] = getattr(fitted_model, attr_to_save_separately)
    model_results_to_save_separately['ses_race_results'] = ses_race_results

    if SAVE_MODEL_RESULTS_SEPARATELY:
        # Save some smaller model results for quick(er) loading. 
        # For really fast stuff, like losses (numerical results only) we store separately.
        print("Saving model results...")
        file = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'wb')
        pickle.dump(model_results_to_save_separately, file)
        file.close()
    
    #----- 4c. Save separate descriptive results --------
    # evaluate model fit to cases and save loss separately as well.
    # Everything saved in this data structure should be a summary result - small and fast to load, numbers only!
    print("# ---- Estimating losses - 1. all sample")
    loss_dict = compare_model_vs_real_num_cases(
        nyt_outcomes,
        model_kwargs['min_datetime'],
        model_results=model_results_to_save_separately)
    fast_to_load_results = {'loss_dict':loss_dict}
    
    if train_test_partition is not None:
        print("# ---- Estimating losses - 2 train set")
        train_max = train_test_partition + datetime.timedelta(hours=-1)
        print(train_test_partition)
        print(train_max)
        train_loss_dict = compare_model_vs_real_num_cases(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            compare_end_time = train_max,
            model_results=model_results_to_save_separately)
        fast_to_load_results['train_loss_dict'] = train_loss_dict
        
        print("# ---- Estimating losses - 3 test set")
        test_loss_dict = compare_model_vs_real_num_cases(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            compare_start_time = train_test_partition,
            model_results=model_results_to_save_separately)
        fast_to_load_results['test_loss_dict'] = test_loss_dict
        
        fast_to_load_results['train_test_date_cutoff'] = train_test_partition
        sanity_check_error_metrics(fast_to_load_results)

    fast_to_load_results['clipping_monitor'] = fitted_model.clipping_monitor
    final_infected_fraction = (
        fitted_model.cbg_infected + fitted_model.cbg_removed + fitted_model.cbg_latent).sum(axis=1)/fitted_model.CBG_SIZES.sum()
    fast_to_load_results['final infected fraction'] = final_infected_fraction
    fast_to_load_results['ses_race_summary_results'] = {}
    demographic_group_keys = ['p_black', 'p_white', 'median_household_income']
    for k1 in demographic_group_keys:
        for k2 in ['above_median', 'below_median', 'top_decile', 'bottom_decile', 'above_median_in_own_county', 'below_median_in_own_county']:
            full_key = 'L+I+R, %s_%s' % (k1, k2)
            fast_to_load_results['ses_race_summary_results']['final fraction in state ' + full_key] = ses_race_results[full_key][-1]
    fast_to_load_results['estimated_R0'] = fitted_model.estimated_R0
    fast_to_load_results['intervention_cost'] = fitted_model.INTERVENTION_COST

    for k1 in demographic_group_keys:
        for (top_group, bot_group) in [
            ('above_median', 'below_median'),
            ('top_decile', 'bottom_decile'),
            ('above_median_in_own_county', 'below_median_in_own_county')]:
            top_group_key = f'{k1}_{top_group}'
            bot_group_key = f'{k1}_{bot_group}'
            top_group_LIR_ratio = ((fitted_model.history[top_group_key]['latent'][:, -1] +
                             fitted_model.history[top_group_key]['infected'][:, -1] +
                             fitted_model.history[top_group_key]['removed'][:, -1]) /
                             fitted_model.history[top_group_key]['total_pop'])
            bot_group_LIR_ratio = ((fitted_model.history[bot_group_key]['latent'][:, -1] +
                             fitted_model.history[bot_group_key]['infected'][:, -1] +
                             fitted_model.history[bot_group_key]['removed'][:, -1]) /
                             fitted_model.history[bot_group_key]['total_pop'])
            fast_to_load_results['ses_race_summary_results'][f'{k1}_{bot_group}_over_{top_group}_L+I+R_ratio_fixed'] = bot_group_LIR_ratio / top_group_LIR_ratio

    file = open(os.path.join(FITTED_MODEL_DIR, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()

    #----- 4c. Save kwargs.--------
    data_and_model_kwargs = {'model_kwargs':model_kwargs, 'data_kwargs':data_kwargs, 'experiment_to_run':experiment_to_run}
    file = open(os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'wb')
    pickle.dump(data_and_model_kwargs, file)
    file.close()
    print("Successfully fitted and saved model and data_and_model_kwargs; total time taken %2.3f seconds" % (time.time() - t0))
    return fitted_model


def fit_disease_model_on_real_data(
    d,
    min_datetime,
    max_datetime,
    exogenous_model_kwargs,
    poi_attributes_to_clip,
    preload_poi_visits_list_filename=None,
    poi_cbg_visits_list=None,
    correct_poi_visits=False, #True, <----------------
    multiply_poi_visit_counts_by_census_ratio=False, #True, <----------------
    aggregate_col_to_use='aggregated_cbg_population_adjusted_visitor_home_cbgs',
    cbg_count_cutoff=10,
    cbgs_to_filter_for=None,
    cbg_groups_to_track=None,
    counties_to_track=None,
    include_cbg_prop_out=False,
    model_init_kwargs=None,
    simulation_kwargs=None,
    counterfactual_poi_opening_experiment_kwargs=None,
    counterfactual_retrospective_experiment_kwargs=None,
    return_model_without_fitting=False,
    return_model_and_data_without_fitting=False,
    model_quality_dict=None,
    verbose=True, 
    meta_d=None, 
    msa_name=None):
    """
    Function to prepare data as input for the disease model, and to run the disease simulation on formatted data.

    OVERSIMPLIFIED VERSION of the function in the previous version. Just give us the model output. We'll deal 
    with the rest later down the road.
    
    Parameters
    ----------    
    d : pandas DataFrame
        population data (prev: POI data from SafeGraph)
    min_datetime, max_datetime : DateTime objects
        The first and last hour to simulate.
    exogenous_model_kwargs : dict
        Extra arguments for Model.init_exogenous_variables().
        required keys : [p_sick_at_t0, poi_psi, home_beta]
    poi_attributes_to_clip : dict
        Which POI attributes to clip.
        required keys : [clip_areas, clip_dwell_times, clip_visits]
    preload_poi_visits_list_filename: str
        Name of file from which to load precomputed hourly networks.
    poi_cbg_visits_list : list of sparse matrices
        precomputed hourly networks
    correct_poi_visits : bool
        whether to correct hourly visit counts with dwell time
    multiply_poi_visit_counts_by_census_ratio : bool
        whether to upscale visit counts by a constant factor
        derived using Census data to try to get real visit volumes
    aggregate_col_to_use : str
        the field that holds the aggregated CBG proportions for each POI
    cbg_count_cutoff : int
        the minimum number of POIs a CBG must visit to be included in the model
    cbgs_to_filter_for : list
        only model CBGs in this list
    cbg_groups_to_track : dict
        maps group name to CBGs, will track their disease trajectories during simulation
    counties_to_track : list
        names of counties, will track their disease trajectories during simulation
    include_cbg_prop_out : bool
        whether to adjust the POI-CBG network based on Social Distancing Metrics (SDM);
        should only be used if precomputed poi_cbg_visits_list is not in use
    model_init_kwargs : dict
        extra arguments for initializing Model
    simulation_kwargs : dict
        extra arguments for Model.simulate_disease_spread()
    counterfactual_poi_opening_experiment_kwargs : dict
        arguments for POI category reopening experiments
    counterfactual_retrospective_experiment_kwargs : dict
        arguments for counterfactual mobility reduction experiment
    meta_d : pandas DataFrame
        POI metadata (municipality and category).
    msa_name: string
        Name of the territory for analysis
    """
    # Argument checks
    assert min_datetime <= max_datetime
    assert all([k in exogenous_model_kwargs for k in ['poi_psi', 'home_beta', 'p_sick_at_t0']])
    assert all([k in poi_attributes_to_clip for k in ['clip_areas', 'clip_dwell_times', 'clip_visits']])
    assert aggregate_col_to_use in [
        'aggregated_cbg_population_adjusted_visitor_home_cbgs',
        'aggregated_visitor_home_cbgs']
    if cbg_groups_to_track is None:
        cbg_groups_to_track = {}
    if model_init_kwargs is None:
        model_init_kwargs = {}
    if simulation_kwargs is None:
        simulation_kwargs = {}
    assert not (return_model_without_fitting and return_model_and_data_without_fitting)

    t0 = time.time()  

    # ALWAYS Load visit networks
    if preload_poi_visits_list_filename is not None:
        f = open(preload_poi_visits_list_filename, 'rb')
        poi_cbg_visits_list = pickle.load(f)
        f.close()
    else:
        print("No mobility provided; exiting.")
        return
    poi_cbg_visits_list = [m.transpose() for m in poi_cbg_visits_list]

    # GB 05/05: moved here from below since we modify mobility data when testing the 2019 counterfactual
    
    if msa_name is None:
       msa_name = BIGGEST_MSAS[0]

    ## NEW_V2
    print('1. Processing population data...')    
    if counterfactual_retrospective_experiment_kwargs is not None:
        # must have one but not all of these arguments
        t1 = 'distancing_degree' in counterfactual_retrospective_experiment_kwargs
        t2 = 'shift_in_days' in counterfactual_retrospective_experiment_kwargs
        t3 = 'rescale_mobility' in counterfactual_retrospective_experiment_kwargs
        t4 = 'grid_values' in counterfactual_retrospective_experiment_kwargs
        assert (t1 + t2 + t3 + t4) == 1

        if poi_cbg_visits_list is None:
            raise Exception('Retrospective experiments are only implemented for when poi_cbg_visits_list is precomputed')
        
        if 'distancing_degree' in counterfactual_retrospective_experiment_kwargs:
            distancing_degree = counterfactual_retrospective_experiment_kwargs['distancing_degree']            
            poi_cbg_visits_list = apply_distancing_degree(poi_cbg_visits_list, distancing_degree, msa_name)
            print('Modified poi_cbg_visits_list for retrospective experiment: distancing_degree = %s.' % distancing_degree)            
            
            if distancing_degree == -1:
                # Load population data
                population_2020 = d                
                pop_path = manage_exporting_paths(wave, data_type, year=2019)[1]
                population_filename = os.path.join(
                    pop_path, f"{msa_name}.csv") #HARDCODING
                assert os.path.exists(population_filename)
                population_2019 = pd.read_csv(population_filename)

                # Check overlap and obtain list of common territories
                a = set(population_2020['municipality']) 
                b = set(population_2019['municipality'])
                keep_pop_2020 = population_2020['municipality'].isin(a-b) == False
                keep_pop_2019 = population_2019['municipality'].isin(b-a) == False
                assert population_2020[keep_pop_2020]['municipality'].tolist() == population_2019[keep_pop_2019]['municipality'].tolist()
                
                # Filter and overwrite the demographic reference file
                d = population_2020[keep_pop_2020].reset_index(drop=True)
            
            # NEW_rebuttal_20240106
            if distancing_degree == -2:
                # Load population data
                population_2020 = d                
                pop_path = manage_exporting_paths(wave, data_type, year=2021)[1]
                population_filename = os.path.join(
                    pop_path, f"{msa_name}.csv") #HARDCODING
                assert os.path.exists(population_filename)
                population_2021 = pd.read_csv(population_filename)

                # Check overlap and obtain list of common territories
                a = set(population_2020['municipality']) 
                b = set(population_2021['municipality'])
                keep_pop_2020 = population_2020['municipality'].isin(a-b) == False
                keep_pop_2021 = population_2021['municipality'].isin(b-a) == False
                assert population_2020[keep_pop_2020]['municipality'].tolist() == population_2021[
                    keep_pop_2021]['municipality'].tolist()
                
                # Filter and overwrite the demographic reference file
                d = population_2020[keep_pop_2020].reset_index(drop=True)
        
        elif 'shift_in_days' in counterfactual_retrospective_experiment_kwargs:
            shift_in_days = counterfactual_retrospective_experiment_kwargs['shift_in_days']
            poi_cbg_visits_list = apply_shift_in_days(poi_cbg_visits_list, shift_in_days)
            print('Modified poi_cbg_visits_list for retrospective experiment: shifted by %d days.' % shift_in_days) 

        elif 'rescale_mobility' in counterfactual_retrospective_experiment_kwargs:
            rescale_by = counterfactual_retrospective_experiment_kwargs['rescale_mobility']
            poi_cbg_visits_list = apply_mobility_rescaling(poi_cbg_visits_list, rescale_by)
            print('Modified poi_cbg_visits_list for retrospective experiment: rescaled by %d.' % rescale_by)    

        elif 'grid_values' in counterfactual_retrospective_experiment_kwargs:
            grid_values = counterfactual_retrospective_experiment_kwargs['grid_values']            
            poi_cbg_visits_list = apply_category_grid(
                msa_name, poi_cbg_visits_list, grid_values)
            
            print(f'Modified poi_cbg_visits_list for retrospective experiment.')
            print(f'Target_categories = {grid_values}.')

            # Load population data
            population_2020 = d                
            pop_path = manage_exporting_paths(wave, data_type, year=2019)[1]
            population_filename = os.path.join(
                pop_path, f"{ msa_name}.csv") #HARDCODING
            assert os.path.exists(population_filename)
            population_2019 = pd.read_csv(population_filename)

            # Check overlap and obtain list of common territories
            a = set(population_2020['municipality']) 
            b = set(population_2019['municipality'])
            keep_pop_2020 = population_2020['municipality'].isin(a-b) == False
            keep_pop_2019 = population_2019['municipality'].isin(b-a) == False
            assert population_2020[keep_pop_2020]['municipality'].tolist() == population_2019[keep_pop_2019]['municipality'].tolist()
            
            # Filter and overwrite the demographic reference file
            d = population_2020[keep_pop_2020].reset_index(drop=True)


    cbg_sizes = d['population'].values
    assert np.sum(np.isnan(cbg_sizes)) == 0

    if verbose:
        print('CBGs: median population size = %d, sum of population sizes = %d' %
          (np.median(cbg_sizes), np.sum(cbg_sizes)))

    print('2. Processing SafeGraph data...')
    # get hour column strings
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    if poi_cbg_visits_list is not None:
        assert len(poi_cbg_visits_list) == len(all_hours)
        hour_cols = all_hours 
    
    ## Skip test because d is different
    #hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    #assert(all([col in d.columns for col in hour_cols]))
    
    # print("Found %d hours in all (%s to %s) -> %d hourly visits" % (len(all_hours),
    #      get_datetime_hour_as_string(min_datetime),
    #      get_datetime_hour_as_string(max_datetime),
    #      np.nansum(d[hour_cols].values)))

    print("Found %d hours in all (%s to %s)" % (len(all_hours),
         get_datetime_hour_as_string(min_datetime),
         get_datetime_hour_as_string(max_datetime)))
    
    ## completely useless
    #all_states = sorted(list(set(d['region'].dropna())))
    all_states = None

    ## DO NOT aggregate median_dwell time over weeks because we do not correct for median dwell
    # weekly_median_dwell_pattern = re.compile('2020-\d\d-\d\d.median_dwell')
    # median_dwell_cols = [col for col in d.columns if re.match(weekly_median_dwell_pattern, col)]
    # print('Aggregating median_dwell from %s to %s' % (median_dwell_cols[0], median_dwell_cols[-1]))
    # # note: this may trigger "RuntimeWarning: All-NaN slice encountered" if a POI has all nans for median_dwell;
    # # this is not a problem and will be addressed in apply_percentile_based_clipping_to_msa_df
    # avg_dwell_times = d[median_dwell_cols].median(axis=1).values
    # d['avg_median_dwell'] = avg_dwell_times

    # DO NOT clip before dropping data so we have more POIs as basis for percentiles
    # this will also drop POIs whose sub and top categories are too small for clipping    
    #poi_attributes_to_clip = poi_attributes_to_clip.copy()  # copy in case we need to modify    
    if poi_cbg_visits_list is not None:
        poi_attributes_to_clip['clip_visits'] = False
        poi_attributes_to_clip['clip_areas'] = False
        poi_attributes_to_clip['clip_dwell_times'] = False
        print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be clipping hourly visits in dataframe')
    
    # if poi_attributes_to_clip['clip_areas'] or poi_attributes_to_clip['clip_dwell_times'] or poi_attributes_to_clip['clip_visits']:
    #     d, categories_to_clip, cols_to_clip, thresholds, medians = clip_poi_attributes_in_msa_df(
    #         d, min_datetime, max_datetime, **poi_attributes_to_clip)
    #     print('After clipping, %i POIs' % len(d))

    # DO NOT remove POIs with missing data
    # d = d.dropna(subset=hour_cols)
    # if verbose: print("After dropping for missing hourly visits, %i POIs" % len(d))
    # d = d.loc[d[aggregate_col_to_use].map(lambda x:len(x.keys()) > 0)]
    # if verbose: print("After dropping for missing CBG home data, %i POIs" % len(d))
    # d = d.dropna(subset=['avg_median_dwell'])
    # if verbose: print("After dropping for missing avg_median_dwell, %i POIs" % len(d))

    # reindex CBGs
    # an array of dicts; each dict represents CBG distribution for POI
    # poi_cbg_proportions = d[aggregate_col_to_use].values  
    # all_cbgs = [a for b in poi_cbg_proportions for a in b.keys()]
    # cbg_counts = Counter(all_cbgs).most_common()
    # # only keep CBGs that have visited at least this many POIs
    # all_unique_cbgs = [cbg for cbg, count in cbg_counts if count >= cbg_count_cutoff]
    
    # if cbgs_to_filter_for is not None:
    #     print("Prior to filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
    #     all_unique_cbgs = [a for a in all_unique_cbgs if a in cbgs_to_filter_for]
    #     print("After filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))

    # # order CBGs lexicographically
    # all_unique_cbgs = sorted(all_unique_cbgs)
    # N = len(all_unique_cbgs)
    # if verbose: print("After dropping CBGs that appear in < %i POIs, %i CBGs (%2.1f%%)" %
    #       (cbg_count_cutoff, N, 100.*N/len(cbg_counts)))
    # cbgs_to_idxs = dict(zip(all_unique_cbgs, range(N)))

    # # convert data structures with CBG names to CBG indices
    # poi_cbg_proportions_int_keys = []
    # kept_poi_idxs = []
    # E = 0   # number of connected POI-CBG pairs
    # for poi_idx, old_dict in enumerate(poi_cbg_proportions):
    #     new_dict = {}
    #     for string_key in old_dict:
    #         if string_key in cbgs_to_idxs:
    #             int_key = cbgs_to_idxs[string_key]
    #             new_dict[int_key] = old_dict[string_key]
    #             E += 1
    #     if len(new_dict) > 0:
    #         poi_cbg_proportions_int_keys.append(new_dict)
    #         kept_poi_idxs.append(poi_idx)            
    # M = len(kept_poi_idxs)
    # if verbose:
    #     print('Dropped %d POIs whose visitors all come from dropped CBGs' %
    #           (len(poi_cbg_proportions) - M))
    
    M, N = poi_cbg_visits_list[0].shape      
    E = max([m.sum().sum() for m in poi_cbg_visits_list])    
    print('FINAL: number of CBGs (N) = %d, number of POIs (M) = %d' % (N, M))
    print('Num connected POI-CBG pairs (E) = %d, network density (E/N) = %.3f' %
          (E, E / N))  # avg num POIs per CBG
    
    # if poi_cbg_visits_list is not None:
    #     expected_M, expected_N = poi_cbg_visits_list[0].shape
    #     assert M == expected_M
    #     assert N == expected_N

    # cbg_idx_groups_to_track = {}
    # for group in cbg_groups_to_track:
    #     cbg_idx_groups_to_track[group] = [
    #         cbgs_to_idxs[a] for a in cbg_groups_to_track[group] if a in cbgs_to_idxs]
    #     if verbose: print(
    #         f'{len(cbg_groups_to_track[group])} CBGs in {group} -> matched {len(cbg_idx_groups_to_track[group])} ({(len(cbg_idx_groups_to_track[group]) / len(cbg_groups_to_track[group])):.3f})')

    # get POI-related variables
    # TODO: avoid loading dwell time if correct_poi_visits is False
    #d = d.iloc[kept_poi_idxs]
    
    #poi_subcategory_types = d['sub_category'].values
    poi_subcategory_types = None
    
    #poi_areas = d['safegraph_computed_area_in_square_feet'].values
    # NEW_V2: 1076.39 square feet is 100 square meters
    #poi_areas = np.ones(M)*1076.39    
    #poi_areas = np.random.normal(1076.39, 250, M) # added F.P. 19 Nov 2021
    
    # NEW_V4: 
    # if ('square_feet_mean' in exogenous_model_kwargs) and ('square_feet_sigma' in exogenous_model_kwargs):
    #     poi_areas = np.random.normal(
    #         exogenous_model_kwargs['square_feet_mean'],
    #         exogenous_model_kwargs['square_feet_sigma'],        
    #         M) # added GB 20220901

    # NEW: added GB 20/05/23
    if 'area_model' in exogenous_model_kwargs:
        if exogenous_model_kwargs['area_model'] == 'normal':
            poi_areas = np.random.normal(
                exogenous_model_kwargs['square_feet_mean'],
                exogenous_model_kwargs['square_feet_sigma'],        
            M)        
        elif exogenous_model_kwargs['area_model'] == 'constant':
            poi_areas = np.ones(M) * exogenous_model_kwargs['square_feet_mean']
        elif exogenous_model_kwargs['area_model'] == 'uniform':
            poi_areas = np.random.uniform(861,1291,M)
        else:
            print("ERROR: area model not recognized.")
    else:
        print("WARNING: area model was not passed. Using default values.")
        exogenous_model_kwargs['square_feet_mean'] = 1076.39, # added GB 20220923
        exogenous_model_kwargs['square_feet_sigma'] = 240, # added GB 20220923
        poi_areas = np.random.normal(
            exogenous_model_kwargs['square_feet_mean'],
            exogenous_model_kwargs['square_feet_sigma'],        
            M) # added GB 20220901

    # NEW_V2: 38 is the median dwell time in the SafeGraph data
    # if we are making an error we are making an underestimation one
    # dwell time seems an exponential distribution with an heavy right tail
    # with mean value is almost twice the median one
    
    #poi_dwell_times = np.ones(M)*38       
    # poi_dwell_times = np.random.exponential(exogenous_model_kwargs['mean_dwell'],M)
    
    if 'dwell_model' in exogenous_model_kwargs:
        if exogenous_model_kwargs['dwell_model'] == 'exponential':
            poi_dwell_times = np.random.exponential(
                exogenous_model_kwargs['mean_dwell'],M)
        elif exogenous_model_kwargs['dwell_model'] == 'lognormal':
            dwell_distr = lognorm(
                loc=3.78,s=1,
                scale = exogenous_model_kwargs['mean_dwell'])
            poi_dwell_times = dwell_distr.rvs(M)
        else:
            print("ERROR: dwell model not recognized.") 
    else:
        print("WARNING: dwell model was not passed. Using default values.")
        poi_dwell_times = np.random.exponential(exogenous_model_kwargs['mean_dwell'],M)
    
    # del exogenous_model_kwargs['mean_dwell'] ## AGGIUNTA FP 23/12	
    
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    print('Dwell time correction factors: mean = %.2f, min = %.2f, max = %.2f' %
          (np.mean(poi_dwell_time_correction_factors), 
          min(poi_dwell_time_correction_factors), 
          max(poi_dwell_time_correction_factors)))
    
    #poi_time_counts = d[hour_cols].values
    poi_time_counts = np.ones([N,len(all_hours)])

    ## DO NOT correct POI visits 
    # if correct_poi_visits:
    #     if poi_cbg_visits_list is not None:
    #         print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be applying correction to hourly visits in dataframe')
    #     else:
    #         print('Correcting POI hourly visit vectors...')
    #         new_poi_time_counts = []
    #         for i, (visit_vector, dwell_time) in enumerate(list(zip(poi_time_counts, poi_dwell_times))):
    #             new_poi_time_counts.append(correct_visit_vector_by_dwell(visit_vector, dwell_time))
    #         poi_time_counts = np.array(new_poi_time_counts)
    #         d[hour_cols] = poi_time_counts
    #         new_hourly_visit_count = np.sum(poi_time_counts)
    #         print('After correcting, %.2f hourly visits' % new_hourly_visit_count)

    # get CBG-related variables from census data
    # print('2. Processing ACS data...')
    # acs_d = helper.load_and_reconcile_multiple_acs_data()
    # cbgs_to_census_pops = dict(zip(
    #     acs_d['census_block_group'].values,
    #     acs_d['total_cbg_population_2018_1YR'].values))  # use most recent population data
    # cbg_sizes = np.array([cbgs_to_census_pops[a] for a in all_unique_cbgs])
    

    # if multiply_poi_visit_counts_by_census_ratio:
    #     # Get overall undersampling factor.
    #     # Basically we take ratio of ACS US population to SafeGraph population in Feb 2020.
    #     # SafeGraph thinks this is reasonable.
    #     # https://safegraphcovid19.slack.com/archives/C0109NPA543/p1586801883190800?thread_ts=1585770817.335800&cid=C0109NPA543
    #     total_us_population_in_50_states_plus_dc = acs_d.loc[
    #         acs_d['state_code'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC), 'total_cbg_population_2018_1YR'].sum()
    #     safegraph_visitor_count_df = pd.read_csv(PATH_TO_OVERALL_HOME_PANEL_SUMMARY)
    #     # safegraph_visitor_count = safegraph_visitor_count_df.loc[
    #     #     safegraph_visitor_count_df['state'] == 'ALL_STATES', 'num_unique_visitors'].iloc[0]
    #     safegraph_visitor_count = safegraph_visitor_count_df.loc[
    #         safegraph_visitor_count_df['region'] == 'ALL_US', 'num_unique_visitors'].iloc[0]

    #     # remove a few safegraph visitors from non-US states.
    #     two_letter_codes_for_states = set([a.lower() for a in codes_to_states if codes_to_states[a] in JUST_50_STATES_PLUS_DC])
    #     #safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['state'].map(
    #         # lambda x:x not in two_letter_codes_for_states and x != 'ALL_STATES'), 'num_unique_visitors'].sum()
    #     safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['region'].map(            
    #         lambda x:x not in two_letter_codes_for_states and x != 'ALL_US'), 'num_unique_visitors'].sum()
    #     if verbose:
    #         print("Removing %2.3f%% of people from SafeGraph count who are not in 50 states or DC" %
    #             (100. * safegraph_visitor_count_to_non_states/safegraph_visitor_count))
    #     safegraph_visitor_count = safegraph_visitor_count - safegraph_visitor_count_to_non_states
    #     correction_factor = 1. * total_us_population_in_50_states_plus_dc / safegraph_visitor_count
    #     if verbose:
    #         print("Total US population from ACS: %i; total safegraph visitor count: %i; correction factor for POI visits is %2.3f" %
    #             (total_us_population_in_50_states_plus_dc,
    #             safegraph_visitor_count,
    #             correction_factor))
    #     poi_time_counts = poi_time_counts * correction_factor

    # if counties_to_track is not None:
    #     print('Found %d counties to track...' % len(counties_to_track))
    #     county2cbgs = {}
    #     for county in counties_to_track:
    #         county_cbgs = acs_d[acs_d['county_code'] == county]['census_block_group'].values
    #         orig_len = len(county_cbgs)
    #         county_cbgs = sorted(set(county_cbgs).intersection(set(all_unique_cbgs)))
    #         if orig_len > 0:
    #             coverage = len(county_cbgs) / orig_len
    #             if coverage < 0.8:
    #                 print('Low coverage warning: only modeling %d/%d (%.1f%%) of the CBGs in %s' %
    #                       (len(county_cbgs), orig_len, 100. * coverage, county))
    #         if len(county_cbgs) > 0:
    #             county_cbg_idx = np.array([cbgs_to_idxs[a] for a in county_cbgs])
    #             county2cbgs[county] = (county_cbgs, county_cbg_idx)
    #     print('Modeling CBGs from %d of the counties' % len(county2cbgs))
    # else:
    #     county2cbgs = None

    # # turn off warnings temporarily so that using > or <= on np.nan does not cause warnings
    # np.warnings.filterwarnings('ignore')
    # cbg_idx_to_track = set(range(N))  # include all CBGs
    # for attribute in ['p_black', 'p_white', 'median_household_income']:
    #     attr_col_name = '%s_2017_5YR' % attribute  # using 5-year ACS data for attributes bc less noisy
    #     assert attr_col_name in acs_d.columns
    #     mapper_d = dict(zip(acs_d['census_block_group'].values, acs_d[attr_col_name].values))
    #     attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in all_unique_cbgs])
    #     non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
    #     median_cutoff = np.median(non_nan_vals)
    #     if verbose:
    #         print("Attribute %s: was able to compute for %2.1f%% out of %i CBGs, median is %2.3f" %
    #             (attribute, 100. * len(non_nan_vals) / len(cbg_idx_to_track),
    #              len(cbg_idx_to_track), median_cutoff))

    #     cbg_idx_groups_to_track[f'{attribute}_above_median'] = list(set(np.where(attribute_vals > median_cutoff)[0]).intersection(cbg_idx_to_track))
    #     cbg_idx_groups_to_track[f'{attribute}_below_median'] = list(set(np.where(attribute_vals <= median_cutoff)[0]).intersection(cbg_idx_to_track))

    #     top_decile = scoreatpercentile(non_nan_vals, 90)
    #     bottom_decile = scoreatpercentile(non_nan_vals, 10)
    #     cbg_idx_groups_to_track[f'{attribute}_top_decile'] = list(set(np.where(attribute_vals >= top_decile)[0]).intersection(cbg_idx_to_track))
    #     cbg_idx_groups_to_track[f'{attribute}_bottom_decile'] = list(set(np.where(attribute_vals <= bottom_decile)[0]).intersection(cbg_idx_to_track))

    #     if county2cbgs is not None:
    #         above_median_in_county = []
    #         below_median_in_county = []
    #         for county in county2cbgs:
    #             county_cbgs, cbg_idx = county2cbgs[county]
    #             attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in county_cbgs])
    #             non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
    #             median_cutoff = np.median(non_nan_vals)
    #             above_median_idx = cbg_idx[np.where(attribute_vals > median_cutoff)[0]]
    #             above_median_idx = list(set(above_median_idx).intersection(cbg_idx_to_track))
    #             above_median_in_county.extend(above_median_idx)
    #             below_median_idx = cbg_idx[np.where(attribute_vals <= median_cutoff)[0]]
    #             below_median_idx = list(set(below_median_idx).intersection(cbg_idx_to_track))
    #             below_median_in_county.extend(below_median_idx)
    #         cbg_idx_groups_to_track[f'{attribute}_above_median_in_own_county'] = above_median_in_county
    #         cbg_idx_groups_to_track[f'{attribute}_below_median_in_own_county'] = below_median_in_county
    # np.warnings.resetwarnings()

    # if include_cbg_prop_out:
    #     model_days = helper.list_datetimes_in_range(min_datetime, max_datetime)
    #     cols_to_keep = ['%s.%s.%s' % (dt.year, dt.month, dt.day) for dt in model_days]
    #     print('Giving model prop out for %s to %s' % (cols_to_keep[0], cols_to_keep[-1]))
    #     assert((len(cols_to_keep) * 24) == len(hour_cols))
        
    #     print('Loading Social Distancing Metrics and computing CBG prop out per day: warning, this could take a while...')
    #     sdm_mdl = helper.load_social_distancing_metrics(model_days)
        
    #     cbg_day_prop_out = helper.compute_cbg_day_prop_out(sdm_mdl, all_unique_cbgs)
    #     assert(len(cbg_day_prop_out) == len(all_unique_cbgs))
        
    #     # sort lexicographically, like all_unique_cbgs
    #     cbg_day_prop_out = cbg_day_prop_out.sort_values(by='census_block_group')
    #     assert list(cbg_day_prop_out['census_block_group'].values) == all_unique_cbgs
    #     cbg_day_prop_out = cbg_day_prop_out[cols_to_keep].values
    # else:
    #     cbg_day_prop_out = None

    cbg_day_prop_out = None
    
    # If trying to get the counterfactual where social activity doesn't change, just repeat first week of dataset.
    # We put this in exogenous_model_kwargs because it actually affects how the model runs, not just the data input.
    if 'just_compute_r0' in exogenous_model_kwargs and exogenous_model_kwargs['just_compute_r0']:
        print('Running model to compute r0 -> looping first week visit counts')
        # simulate out 15 weeks just so we are sure all cases are gone.
        max_datetime = min_datetime + datetime.timedelta(hours=(168*15)-1)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        if poi_cbg_visits_list is not None:
            assert len(poi_cbg_visits_list) >= 168  # ensure that we have at least a week to model (7 * 24 = 168)
            new_visits_list = []
            for i in range(168 * 15):
                first_week_idx = i % 168  # map to corresponding hour in first week
                new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
            poi_cbg_visits_list = new_visits_list
            assert len(poi_cbg_visits_list) == len(all_hours)
        else:
            print("No mobility provided; exiting.")
            return
            # assert poi_time_counts.shape[1] >= 168  # ensure that we have at least a week to model
            # first_week = poi_time_counts[:, :168]
            # poi_time_counts = np.tile(first_week, (1, 15))
            # if cbg_day_prop_out is not None:
            #     assert cbg_day_prop_out.shape[1] >= 7
            #     first_week = cbg_day_prop_out[:, :7]
            #     cbg_day_prop_out = np.tile(first_week, (1, 15))
            # assert poi_time_counts.shape[1] == len(all_hours)

    # # If we want to run counterfactual reopening simulations
    intervention_cost = None

    if counterfactual_poi_opening_experiment_kwargs is not None:
        if poi_cbg_visits_list is None:
            raise Exception('Missing poi_cbg_visits_list; reopening experiments should be run with IPF output')
        
        # extend hours if necessary
        extra_weeks_to_simulate = counterfactual_poi_opening_experiment_kwargs['extra_weeks_to_simulate']
        assert extra_weeks_to_simulate >= 0
        orig_num_hours = len(all_hours)
        all_hours = helper.list_hours_in_range(
            min_datetime, max_datetime + datetime.timedelta(hours=168 * extra_weeks_to_simulate))
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        
        # fix moment of intervention
        intervention_datetime = counterfactual_poi_opening_experiment_kwargs['intervention_datetime']
        assert(intervention_datetime in all_hours)
        intervention_hour_idx = all_hours.index(intervention_datetime)
        
        # check categories
        if 'top_category' in counterfactual_poi_opening_experiment_kwargs:
            top_category = counterfactual_poi_opening_experiment_kwargs['top_category']
        else:
            top_category = None

        if 'sub_category' in counterfactual_poi_opening_experiment_kwargs:
            sub_category = counterfactual_poi_opening_experiment_kwargs['sub_category']
        else:
            sub_category = None
        if meta_d is not None:
            poi_categories = meta_d[['top_category', 'sub_category']]
        else:
            print("Error: no POI metadata found to run this experiment.")
            return

        # must have one but not both of these arguments
        assert (('alpha' in counterfactual_poi_opening_experiment_kwargs) + ('full_activity_alpha' in counterfactual_poi_opening_experiment_kwargs)) == 1
        # the original alpha - post-intervention is interpolation between no reopening and full activity
        if 'alpha' in counterfactual_poi_opening_experiment_kwargs:
            alpha = counterfactual_poi_opening_experiment_kwargs['alpha']
            assert alpha >= 0 and alpha <= 1
            # poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(
            #     poi_cbg_visits_list,
            #     poi_categories, poi_areas, all_hours, intervention_hour_idx,
            #     alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=True)
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices_simplified(
                poi_cbg_visits_list, poi_categories,  intervention_hour_idx, alpha, 
                extra_weeks_to_simulate, top_category, sub_category)

        # post-intervention is alpha-percent of full activity (no interpolation)
        else:
            alpha = counterfactual_poi_opening_experiment_kwargs['full_activity_alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(
                poi_cbg_visits_list,
                poi_categories, poi_areas, all_hours, intervention_hour_idx,
                alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=False)

        # should be used in tandem with alpha or full_activity_alpha, since the timeseries is extended
        # in those blocks; this part just caps post-intervention visits to alpha-percent of max capacity
        if 'max_capacity_alpha' in counterfactual_poi_opening_experiment_kwargs:
            max_capacity_alpha = counterfactual_poi_opening_experiment_kwargs['max_capacity_alpha']
            assert max_capacity_alpha >= 0 and max_capacity_alpha <= 1
            poi_visits = np.zeros((M, orig_num_hours))   # num pois x num hours
            for t, poi_cbg_visits in enumerate(poi_cbg_visits_list[:orig_num_hours]):
                poi_visits[:, t] = poi_cbg_visits @ np.ones(N)
            max_per_poi = np.max(poi_visits, axis=1)  # get historical max capacity per POI
            alpha_max_per_poi = np.clip(max_capacity_alpha * max_per_poi, 1e-10, None)  # so that we don't divide by 0
            orig_total_activity = 0
            capped_total_activity = 0
            for t in range(intervention_hour_idx, len(poi_cbg_visits_list)):
                poi_cbg_visits = poi_cbg_visits_list[t]
                num_visits_per_poi = poi_cbg_visits @ np.ones(N)
                orig_total_activity += np.sum(num_visits_per_poi)
                ratio_per_poi = num_visits_per_poi / alpha_max_per_poi
                clipping_idx = ratio_per_poi > 1  # identify which POIs need to be clipped
                poi_multipliers = np.ones(M)
                poi_multipliers[clipping_idx] = 1 / ratio_per_poi[clipping_idx]
                adjusted_poi_cbg_visits = poi_cbg_visits.transpose().multiply(poi_multipliers).transpose().tocsr()
                capped_total_activity += np.sum(adjusted_poi_cbg_visits @ np.ones(N))
                poi_cbg_visits_list[t] = adjusted_poi_cbg_visits
            print('Finished capping visits at %.1f%% of max capacity -> kept %.4f%% of visits' %
                  (100. * max_capacity_alpha, 100 * capped_total_activity / orig_total_activity))
            intervention_cost['total_activity_after_max_capacity_capping'] = capped_total_activity

    # if counterfactual_retrospective_experiment_kwargs is not None: .....


    print('Total time to prep data: %.3fs' % (time.time() - t0))

    # NEW_V2
    poi_cbg_proportions_int_keys = None
    all_unique_cbgs = None
    #cbg_idx_groups_to_track = None
    
    # Track only municipalities in the target province
    cbg_idx_groups_to_track = {
        'nyt': d[d['track']==1].index.tolist(), 
        'all': d[d['track']==1].index.tolist()
    }
    ## Aggiunta FP 23/12 per tracciare una provincia in una regione	
    for col in d.columns:
        if col not in ['municipality', 'population','track']:
            cbg_idx_groups_to_track[col] = d[d[col]==1].index.tolist()
        
            if 'groups_to_track_num_cases_per_poi' in simulation_kwargs:
                simulation_kwargs['groups_to_track_num_cases_per_poi'].append(col)

    cbgs_to_idxs = None

    if INITIALIZE_INFECTED_WITH_OBS:
        nyt_outcomes = get_variables_for_evaluating_province_model(msa_name)
        real_dates, real_cases = get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes)
        cases_df = pd.DataFrame({'date':real_dates, 'cases':real_cases})            
        cases_df['daily'] = get_daily_from_cumulative(cases_df['cases'].values)
        starting_infections = cases_df.loc[cases_df['date']==MIN_DATETIME,'daily']
        
        if starting_infections.empty:                
            initial_infection_constant = None
        else:
            initial_infection_constant = starting_infections.values[0]
    else:
        initial_infection_constant = None
    
    # feed everything into model.
    m = Model(**model_init_kwargs)
    m.init_exogenous_variables(
        poi_cbg_proportions=poi_cbg_proportions_int_keys,#None
        poi_time_counts=poi_time_counts, #all ones
        poi_areas=poi_areas, #all ones
        poi_dwell_time_correction_factors=poi_dwell_time_correction_factors, #None
        cbg_sizes=cbg_sizes,
        all_unique_cbgs=all_unique_cbgs,#None
        cbgs_to_idxs=cbgs_to_idxs,#None
        all_states=all_states,#None
        poi_cbg_visits_list=poi_cbg_visits_list,
        all_hours=all_hours,
        cbg_idx_groups_to_track=cbg_idx_groups_to_track,#None
        cbg_day_prop_out=cbg_day_prop_out,#None
        intervention_cost=intervention_cost,#None
        poi_subcategory_types=poi_subcategory_types,#None
        **exogenous_model_kwargs)
    m.init_endogenous_variables(
        initial_infection_constant=initial_infection_constant)
    if return_model_without_fitting:
        return m
    elif return_model_and_data_without_fitting:
        #m.d = d
        return m
    m.simulate_disease_spread(**simulation_kwargs)
    return m

def fit_disease_model_on_real_data_OLD(d,
    min_datetime,
    max_datetime,
    exogenous_model_kwargs,
    poi_attributes_to_clip,
    preload_poi_visits_list_filename=None,
    poi_cbg_visits_list=None,
    correct_poi_visits=False, #True, <----------------
    multiply_poi_visit_counts_by_census_ratio=False, #True, <----------------
    aggregate_col_to_use='aggregated_cbg_population_adjusted_visitor_home_cbgs',
    cbg_count_cutoff=10,
    cbgs_to_filter_for=None,
    cbg_groups_to_track=None,
    counties_to_track=None,
    include_cbg_prop_out=False,
    model_init_kwargs=None,
    simulation_kwargs=None,
    counterfactual_poi_opening_experiment_kwargs=None,
    counterfactual_retrospective_experiment_kwargs=None,
    return_model_without_fitting=False,
    return_model_and_data_without_fitting=False,
    model_quality_dict=None,
    verbose=True):
    """
    Function to prepare data as input for the disease model, and to run the disease simulation on formatted data.
    
    Parameters
    ----------    
    d : pandas DataFrame
        POI data from SafeGraph.
    min_datetime, max_datetime : DateTime objects
        The first and last hour to simulate.
    exogenous_model_kwargs : dict
        Extra arguments for Model.init_exogenous_variables().
        required keys : [p_sick_at_t0, poi_psi, home_beta]
    poi_attributes_to_clip : dict
        Which POI attributes to clip.
        required keys : [clip_areas, clip_dwell_times, clip_visits]
    preload_poi_visits_list_filename: str
        Name of file from which to load precomputed hourly networks.
    poi_cbg_visits_list : list of sparse matrices
        precomputed hourly networks
    correct_poi_visits : bool
        whether to correct hourly visit counts with dwell time
    multiply_poi_visit_counts_by_census_ratio : bool
        whether to upscale visit counts by a constant factor
        derived using Census data to try to get real visit volumes
    aggregate_col_to_use : str
        the field that holds the aggregated CBG proportions for each POI
    cbg_count_cutoff : int
        the minimum number of POIs a CBG must visit to be included in the model
    cbgs_to_filter_for : list
        only model CBGs in this list
    cbg_groups_to_track : dict
        maps group name to CBGs, will track their disease trajectories during simulation
    counties_to_track : list
        names of counties, will track their disease trajectories during simulation
    include_cbg_prop_out : bool
        whether to adjust the POI-CBG network based on Social Distancing Metrics (SDM);
        should only be used if precomputed poi_cbg_visits_list is not in use
    model_init_kwargs : dict
        extra arguments for initializing Model
    simulation_kwargs : dict
        extra arguments for Model.simulate_disease_spread()
    counterfactual_poi_opening_experiment_kwargs : dict
        arguments for POI category reopening experiments
    counterfactual_retrospective_experiment_kwargs : dict
        arguments for counterfactual mobility reduction experiment
    """
    # Argument checks
    assert min_datetime <= max_datetime
    assert all([k in exogenous_model_kwargs for k in ['poi_psi', 'home_beta', 'p_sick_at_t0']])
    assert all([k in poi_attributes_to_clip for k in ['clip_areas', 'clip_dwell_times', 'clip_visits']])
    assert aggregate_col_to_use in [
        'aggregated_cbg_population_adjusted_visitor_home_cbgs',
        'aggregated_visitor_home_cbgs']
    if cbg_groups_to_track is None:
        cbg_groups_to_track = {}
    if model_init_kwargs is None:
        model_init_kwargs = {}
    if simulation_kwargs is None:
        simulation_kwargs = {}
    assert not (return_model_without_fitting and return_model_and_data_without_fitting)

    # Load visit networks
    if preload_poi_visits_list_filename is not None:
        f = open(preload_poi_visits_list_filename, 'rb')
        poi_cbg_visits_list = pickle.load(f)
        f.close()

    t0 = time.time()    
    print('1. Processing SafeGraph data...')
    # get hour column strings
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    if poi_cbg_visits_list is not None:
        assert len(poi_cbg_visits_list) == len(all_hours)
    
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    assert(all([col in d.columns for col in hour_cols]))
    
    print("Found %d hours in all (%s to %s) -> %d hourly visits" % (len(all_hours),
         get_datetime_hour_as_string(min_datetime),
         get_datetime_hour_as_string(max_datetime),
         np.nansum(d[hour_cols].values)))
    
    # completely useless
    all_states = sorted(list(set(d['region'].dropna())))

    # aggregate median_dwell time over weeks
    weekly_median_dwell_pattern = re.compile('2020-\d\d-\d\d.median_dwell')
    median_dwell_cols = [col for col in d.columns if re.match(weekly_median_dwell_pattern, col)]
    print('Aggregating median_dwell from %s to %s' % (median_dwell_cols[0], median_dwell_cols[-1]))
    # note: this may trigger "RuntimeWarning: All-NaN slice encountered" if a POI has all nans for median_dwell;
    # this is not a problem and will be addressed in apply_percentile_based_clipping_to_msa_df
    avg_dwell_times = d[median_dwell_cols].median(axis=1).values
    d['avg_median_dwell'] = avg_dwell_times

    # clip before dropping data so we have more POIs as basis for percentiles
    # this will also drop POIs whose sub and top categories are too small for clipping
    poi_attributes_to_clip = poi_attributes_to_clip.copy()  # copy in case we need to modify
    if poi_cbg_visits_list is not None:
        poi_attributes_to_clip['clip_visits'] = False
        print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be clipping hourly visits in dataframe')
    if poi_attributes_to_clip['clip_areas'] or poi_attributes_to_clip['clip_dwell_times'] or poi_attributes_to_clip['clip_visits']:
        d, categories_to_clip, cols_to_clip, thresholds, medians = clip_poi_attributes_in_msa_df(
            d, min_datetime, max_datetime, **poi_attributes_to_clip)
        print('After clipping, %i POIs' % len(d))

    # remove POIs with missing data
    d = d.dropna(subset=hour_cols)
    if verbose: print("After dropping for missing hourly visits, %i POIs" % len(d))
    d = d.loc[d[aggregate_col_to_use].map(lambda x:len(x.keys()) > 0)]
    if verbose: print("After dropping for missing CBG home data, %i POIs" % len(d))
    d = d.dropna(subset=['avg_median_dwell'])
    if verbose: print("After dropping for missing avg_median_dwell, %i POIs" % len(d))

    # reindex CBGs
    # an array of dicts; each dict represents CBG distribution for POI
    poi_cbg_proportions = d[aggregate_col_to_use].values  
    all_cbgs = [a for b in poi_cbg_proportions for a in b.keys()]
    cbg_counts = Counter(all_cbgs).most_common()
    # only keep CBGs that have visited at least this many POIs
    all_unique_cbgs = [cbg for cbg, count in cbg_counts if count >= cbg_count_cutoff]
    if cbgs_to_filter_for is not None:
        print("Prior to filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
        all_unique_cbgs = [a for a in all_unique_cbgs if a in cbgs_to_filter_for]
        print("After filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))

    # order CBGs lexicographically
    all_unique_cbgs = sorted(all_unique_cbgs)
    N = len(all_unique_cbgs)
    if verbose: print("After dropping CBGs that appear in < %i POIs, %i CBGs (%2.1f%%)" %
          (cbg_count_cutoff, N, 100.*N/len(cbg_counts)))
    cbgs_to_idxs = dict(zip(all_unique_cbgs, range(N)))

    # convert data structures with CBG names to CBG indices
    poi_cbg_proportions_int_keys = []
    kept_poi_idxs = []
    E = 0   # number of connected POI-CBG pairs
    for poi_idx, old_dict in enumerate(poi_cbg_proportions):
        new_dict = {}
        for string_key in old_dict:
            if string_key in cbgs_to_idxs:
                int_key = cbgs_to_idxs[string_key]
                new_dict[int_key] = old_dict[string_key]
                E += 1
        if len(new_dict) > 0:
            poi_cbg_proportions_int_keys.append(new_dict)
            kept_poi_idxs.append(poi_idx)            
    M = len(kept_poi_idxs)
    if verbose:
        print('Dropped %d POIs whose visitors all come from dropped CBGs' %
              (len(poi_cbg_proportions) - M))
              
    print('FINAL: number of CBGs (N) = %d, number of POIs (M) = %d' % (N, M))
    print('Num connected POI-CBG pairs (E) = %d, network density (E/N) = %.3f' %
          (E, E / N))  # avg num POIs per CBG
    if poi_cbg_visits_list is not None:
        expected_M, expected_N = poi_cbg_visits_list[0].shape
        assert M == expected_M
        assert N == expected_N

    cbg_idx_groups_to_track = {}
    for group in cbg_groups_to_track:
        cbg_idx_groups_to_track[group] = [
            cbgs_to_idxs[a] for a in cbg_groups_to_track[group] if a in cbgs_to_idxs]
        if verbose: print(
            f'{len(cbg_groups_to_track[group])} CBGs in {group} -> matched {len(cbg_idx_groups_to_track[group])} ({(len(cbg_idx_groups_to_track[group]) / len(cbg_groups_to_track[group])):.3f})')

    # get POI-related variables
    # TODO: avoid loading dwell time if correct_poi_visits is False
    d = d.iloc[kept_poi_idxs]
    poi_subcategory_types = d['sub_category'].values
    poi_areas = d['safegraph_computed_area_in_square_feet'].values
    poi_dwell_times = d['avg_median_dwell'].values
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    print('Dwell time correction factors: mean = %.2f, min = %.2f, max = %.2f' %
          (np.mean(poi_dwell_time_correction_factors), min(poi_dwell_time_correction_factors), max(poi_dwell_time_correction_factors)))
    poi_time_counts = d[hour_cols].values
    if correct_poi_visits:
        if poi_cbg_visits_list is not None:
            print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be applying correction to hourly visits in dataframe')
        else:
            print('Correcting POI hourly visit vectors...')
            new_poi_time_counts = []
            for i, (visit_vector, dwell_time) in enumerate(list(zip(poi_time_counts, poi_dwell_times))):
                new_poi_time_counts.append(correct_visit_vector_by_dwell(visit_vector, dwell_time))
            poi_time_counts = np.array(new_poi_time_counts)
            d[hour_cols] = poi_time_counts
            new_hourly_visit_count = np.sum(poi_time_counts)
            print('After correcting, %.2f hourly visits' % new_hourly_visit_count)

    # get CBG-related variables from census data
    print('2. Processing ACS data...')
    acs_d = helper.load_and_reconcile_multiple_acs_data()
    cbgs_to_census_pops = dict(zip(
        acs_d['census_block_group'].values,
        acs_d['total_cbg_population_2018_1YR'].values))  # use most recent population data
    cbg_sizes = np.array([cbgs_to_census_pops[a] for a in all_unique_cbgs])
    assert np.sum(np.isnan(cbg_sizes)) == 0
    if verbose:
        print('CBGs: median population size = %d, sum of population sizes = %d' %
          (np.median(cbg_sizes), np.sum(cbg_sizes)))

    if multiply_poi_visit_counts_by_census_ratio:
        # Get overall undersampling factor.
        # Basically we take ratio of ACS US population to SafeGraph population in Feb 2020.
        # SafeGraph thinks this is reasonable.
        # https://safegraphcovid19.slack.com/archives/C0109NPA543/p1586801883190800?thread_ts=1585770817.335800&cid=C0109NPA543
        total_us_population_in_50_states_plus_dc = acs_d.loc[
            acs_d['state_code'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC), 'total_cbg_population_2018_1YR'].sum()
        safegraph_visitor_count_df = pd.read_csv(PATH_TO_OVERALL_HOME_PANEL_SUMMARY)
        # safegraph_visitor_count = safegraph_visitor_count_df.loc[
        #     safegraph_visitor_count_df['state'] == 'ALL_STATES', 'num_unique_visitors'].iloc[0]
        safegraph_visitor_count = safegraph_visitor_count_df.loc[
            safegraph_visitor_count_df['region'] == 'ALL_US', 'num_unique_visitors'].iloc[0]

        # remove a few safegraph visitors from non-US states.
        two_letter_codes_for_states = set([a.lower() for a in codes_to_states if codes_to_states[a] in JUST_50_STATES_PLUS_DC])
        #safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['state'].map(
            # lambda x:x not in two_letter_codes_for_states and x != 'ALL_STATES'), 'num_unique_visitors'].sum()
        safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['region'].map(            
            lambda x:x not in two_letter_codes_for_states and x != 'ALL_US'), 'num_unique_visitors'].sum()
        if verbose:
            print("Removing %2.3f%% of people from SafeGraph count who are not in 50 states or DC" %
                (100. * safegraph_visitor_count_to_non_states/safegraph_visitor_count))
        safegraph_visitor_count = safegraph_visitor_count - safegraph_visitor_count_to_non_states
        correction_factor = 1. * total_us_population_in_50_states_plus_dc / safegraph_visitor_count
        if verbose:
            print("Total US population from ACS: %i; total safegraph visitor count: %i; correction factor for POI visits is %2.3f" %
                (total_us_population_in_50_states_plus_dc,
                safegraph_visitor_count,
                correction_factor))
        poi_time_counts = poi_time_counts * correction_factor

    if counties_to_track is not None:
        print('Found %d counties to track...' % len(counties_to_track))
        county2cbgs = {}
        for county in counties_to_track:
            county_cbgs = acs_d[acs_d['county_code'] == county]['census_block_group'].values
            orig_len = len(county_cbgs)
            county_cbgs = sorted(set(county_cbgs).intersection(set(all_unique_cbgs)))
            if orig_len > 0:
                coverage = len(county_cbgs) / orig_len
                if coverage < 0.8:
                    print('Low coverage warning: only modeling %d/%d (%.1f%%) of the CBGs in %s' %
                          (len(county_cbgs), orig_len, 100. * coverage, county))
            if len(county_cbgs) > 0:
                county_cbg_idx = np.array([cbgs_to_idxs[a] for a in county_cbgs])
                county2cbgs[county] = (county_cbgs, county_cbg_idx)
        print('Modeling CBGs from %d of the counties' % len(county2cbgs))
    else:
        county2cbgs = None

    # turn off warnings temporarily so that using > or <= on np.nan does not cause warnings
    np.warnings.filterwarnings('ignore')
    cbg_idx_to_track = set(range(N))  # include all CBGs
    for attribute in ['p_black', 'p_white', 'median_household_income']:
        attr_col_name = '%s_2017_5YR' % attribute  # using 5-year ACS data for attributes bc less noisy
        assert attr_col_name in acs_d.columns
        mapper_d = dict(zip(acs_d['census_block_group'].values, acs_d[attr_col_name].values))
        attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in all_unique_cbgs])
        non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
        median_cutoff = np.median(non_nan_vals)
        if verbose:
            print("Attribute %s: was able to compute for %2.1f%% out of %i CBGs, median is %2.3f" %
                (attribute, 100. * len(non_nan_vals) / len(cbg_idx_to_track),
                 len(cbg_idx_to_track), median_cutoff))

        cbg_idx_groups_to_track[f'{attribute}_above_median'] = list(set(np.where(attribute_vals > median_cutoff)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_below_median'] = list(set(np.where(attribute_vals <= median_cutoff)[0]).intersection(cbg_idx_to_track))

        top_decile = scoreatpercentile(non_nan_vals, 90)
        bottom_decile = scoreatpercentile(non_nan_vals, 10)
        cbg_idx_groups_to_track[f'{attribute}_top_decile'] = list(set(np.where(attribute_vals >= top_decile)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_bottom_decile'] = list(set(np.where(attribute_vals <= bottom_decile)[0]).intersection(cbg_idx_to_track))

        if county2cbgs is not None:
            above_median_in_county = []
            below_median_in_county = []
            for county in county2cbgs:
                county_cbgs, cbg_idx = county2cbgs[county]
                attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in county_cbgs])
                non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
                median_cutoff = np.median(non_nan_vals)
                above_median_idx = cbg_idx[np.where(attribute_vals > median_cutoff)[0]]
                above_median_idx = list(set(above_median_idx).intersection(cbg_idx_to_track))
                above_median_in_county.extend(above_median_idx)
                below_median_idx = cbg_idx[np.where(attribute_vals <= median_cutoff)[0]]
                below_median_idx = list(set(below_median_idx).intersection(cbg_idx_to_track))
                below_median_in_county.extend(below_median_idx)
            cbg_idx_groups_to_track[f'{attribute}_above_median_in_own_county'] = above_median_in_county
            cbg_idx_groups_to_track[f'{attribute}_below_median_in_own_county'] = below_median_in_county
    np.warnings.resetwarnings()

    if include_cbg_prop_out:
        model_days = helper.list_datetimes_in_range(min_datetime, max_datetime)
        cols_to_keep = ['%s.%s.%s' % (dt.year, dt.month, dt.day) for dt in model_days]
        print('Giving model prop out for %s to %s' % (cols_to_keep[0], cols_to_keep[-1]))
        assert((len(cols_to_keep) * 24) == len(hour_cols))
        
        print('Loading Social Distancing Metrics and computing CBG prop out per day: warning, this could take a while...')
        sdm_mdl = helper.load_social_distancing_metrics(model_days)
        
        cbg_day_prop_out = helper.compute_cbg_day_prop_out(sdm_mdl, all_unique_cbgs)
        assert(len(cbg_day_prop_out) == len(all_unique_cbgs))
        
        # sort lexicographically, like all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out.sort_values(by='census_block_group')
        assert list(cbg_day_prop_out['census_block_group'].values) == all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out[cols_to_keep].values
    else:
        cbg_day_prop_out = None

    # If trying to get the counterfactual where social activity doesn't change, just repeat first week of dataset.
    # We put this in exogenous_model_kwargs because it actually affects how the model runs, not just the data input.
    if 'just_compute_r0' in exogenous_model_kwargs and exogenous_model_kwargs['just_compute_r0']:
        print('Running model to compute r0 -> looping first week visit counts')
        # simulate out 15 weeks just so we are sure all cases are gone.
        max_datetime = min_datetime + datetime.timedelta(hours=(168*15)-1)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        if poi_cbg_visits_list is not None:
            assert len(poi_cbg_visits_list) >= 168  # ensure that we have at least a week to model (7 * 24 = 168)
            new_visits_list = []
            for i in range(168 * 15):
                first_week_idx = i % 168  # map to corresponding hour in first week
                new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
            poi_cbg_visits_list = new_visits_list
            assert len(poi_cbg_visits_list) == len(all_hours)
        else:
            assert poi_time_counts.shape[1] >= 168  # ensure that we have at least a week to model
            first_week = poi_time_counts[:, :168]
            poi_time_counts = np.tile(first_week, (1, 15))
            if cbg_day_prop_out is not None:
                assert cbg_day_prop_out.shape[1] >= 7
                first_week = cbg_day_prop_out[:, :7]
                cbg_day_prop_out = np.tile(first_week, (1, 15))
            assert poi_time_counts.shape[1] == len(all_hours)

    # If we want to run counterfactual reopening simulations
    intervention_cost = None
    if counterfactual_poi_opening_experiment_kwargs is not None:
        if poi_cbg_visits_list is None:
            raise Exception('Missing poi_cbg_visits_list; reopening experiments should be run with IPF output')
        extra_weeks_to_simulate = counterfactual_poi_opening_experiment_kwargs['extra_weeks_to_simulate']
        assert extra_weeks_to_simulate >= 0
        orig_num_hours = len(all_hours)
        all_hours = helper.list_hours_in_range(min_datetime, max_datetime + datetime.timedelta(hours=168 * extra_weeks_to_simulate))
        print("Extending time period; simulation now ends at %s (%d hours)" % (max(all_hours), len(all_hours)))
        intervention_datetime = counterfactual_poi_opening_experiment_kwargs['intervention_datetime']
        assert(intervention_datetime in all_hours)
        intervention_hour_idx = all_hours.index(intervention_datetime)
        if 'top_category' in counterfactual_poi_opening_experiment_kwargs:
            top_category = counterfactual_poi_opening_experiment_kwargs['top_category']
        else:
            top_category = None
        if 'sub_category' in counterfactual_poi_opening_experiment_kwargs:
            sub_category = counterfactual_poi_opening_experiment_kwargs['sub_category']
        else:
            sub_category = None
        poi_categories = d[['top_category', 'sub_category']]

        # must have one but not both of these arguments
        assert (('alpha' in counterfactual_poi_opening_experiment_kwargs) + ('full_activity_alpha' in counterfactual_poi_opening_experiment_kwargs)) == 1
        # the original alpha - post-intervention is interpolation between no reopening and full activity
        if 'alpha' in counterfactual_poi_opening_experiment_kwargs:
            alpha = counterfactual_poi_opening_experiment_kwargs['alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(
                poi_cbg_visits_list,
                poi_categories, poi_areas, all_hours, intervention_hour_idx,
                alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=True)
        # post-intervention is alpha-percent of full activity (no interpolation)
        else:
            alpha = counterfactual_poi_opening_experiment_kwargs['full_activity_alpha']
            assert alpha >= 0 and alpha <= 1
            poi_cbg_visits_list, intervention_cost = apply_interventions_to_poi_cbg_matrices(
                poi_cbg_visits_list,
                poi_categories, poi_areas, all_hours, intervention_hour_idx,
                alpha, extra_weeks_to_simulate, top_category, sub_category, interpolate=False)

        # should be used in tandem with alpha or full_activity_alpha, since the timeseries is extended
        # in those blocks; this part just caps post-intervention visits to alpha-percent of max capacity
        if 'max_capacity_alpha' in counterfactual_poi_opening_experiment_kwargs:
            max_capacity_alpha = counterfactual_poi_opening_experiment_kwargs['max_capacity_alpha']
            assert max_capacity_alpha >= 0 and max_capacity_alpha <= 1
            poi_visits = np.zeros((M, orig_num_hours))   # num pois x num hours
            for t, poi_cbg_visits in enumerate(poi_cbg_visits_list[:orig_num_hours]):
                poi_visits[:, t] = poi_cbg_visits @ np.ones(N)
            max_per_poi = np.max(poi_visits, axis=1)  # get historical max capacity per POI
            alpha_max_per_poi = np.clip(max_capacity_alpha * max_per_poi, 1e-10, None)  # so that we don't divide by 0
            orig_total_activity = 0
            capped_total_activity = 0
            for t in range(intervention_hour_idx, len(poi_cbg_visits_list)):
                poi_cbg_visits = poi_cbg_visits_list[t]
                num_visits_per_poi = poi_cbg_visits @ np.ones(N)
                orig_total_activity += np.sum(num_visits_per_poi)
                ratio_per_poi = num_visits_per_poi / alpha_max_per_poi
                clipping_idx = ratio_per_poi > 1  # identify which POIs need to be clipped
                poi_multipliers = np.ones(M)
                poi_multipliers[clipping_idx] = 1 / ratio_per_poi[clipping_idx]
                adjusted_poi_cbg_visits = poi_cbg_visits.transpose().multiply(poi_multipliers).transpose().tocsr()
                capped_total_activity += np.sum(adjusted_poi_cbg_visits @ np.ones(N))
                poi_cbg_visits_list[t] = adjusted_poi_cbg_visits
            print('Finished capping visits at %.1f%% of max capacity -> kept %.4f%% of visits' %
                  (100. * max_capacity_alpha, 100 * capped_total_activity / orig_total_activity))
            intervention_cost['total_activity_after_max_capacity_capping'] = capped_total_activity

    if counterfactual_retrospective_experiment_kwargs is not None:
        # must have one but not both of these arguments
        assert (('distancing_degree' in counterfactual_retrospective_experiment_kwargs) + ('shift_in_days' in counterfactual_retrospective_experiment_kwargs)) == 1
        if poi_cbg_visits_list is None:
            raise Exception('Retrospective experiments are only implemented for when poi_cbg_visits_list is precomputed')
        if 'distancing_degree' in counterfactual_retrospective_experiment_kwargs:
            distancing_degree = counterfactual_retrospective_experiment_kwargs['distancing_degree']
            poi_cbg_visits_list = apply_distancing_degree(poi_cbg_visits_list, distancing_degree)
            print('Modified poi_cbg_visits_list for retrospective experiment: distancing_degree = %s.' % distancing_degree)
        else:
            shift_in_days = counterfactual_retrospective_experiment_kwargs['shift_in_days']
            poi_cbg_visits_list = apply_shift_in_days(poi_cbg_visits_list, shift_in_days)
            print('Modified poi_cbg_visits_list for retrospective experiment: shifted by %d days.' % shift_in_days)

    print('Total time to prep data: %.3fs' % (time.time() - t0))

    # feed everything into model.
    m = Model(**model_init_kwargs)
    m.init_exogenous_variables(
        poi_cbg_proportions=poi_cbg_proportions_int_keys,
        poi_time_counts=poi_time_counts,
        poi_areas=poi_areas,
        poi_dwell_time_correction_factors=poi_dwell_time_correction_factors,
        cbg_sizes=cbg_sizes,
        all_unique_cbgs=all_unique_cbgs,
        cbgs_to_idxs=cbgs_to_idxs,
        all_states=all_states,
        poi_cbg_visits_list=poi_cbg_visits_list,
        all_hours=all_hours,
        cbg_idx_groups_to_track=cbg_idx_groups_to_track,
        cbg_day_prop_out=cbg_day_prop_out,
        intervention_cost=intervention_cost,
        poi_subcategory_types=poi_subcategory_types,
        **exogenous_model_kwargs)
    m.init_endogenous_variables()
    if return_model_without_fitting:
        return m
    elif return_model_and_data_without_fitting:
        m.d = d
        return m
    m.simulate_disease_spread(**simulation_kwargs)
    return m

def correct_visit_vector_by_dwell(v, median_dwell_in_minutes):
    """
    NEW: it was called correct_visit_vector before.
    Given an original hourly visit vector v and a dwell time in minutes,
    return a new hourly visit vector which accounts for spillover.

    Rationale: there may be overlapping among visiting hours, hence 
    visitors have to be attributed to a certain hour.
    """
    v = np.array(v)
    d = median_dwell_in_minutes/60.
    new_v = v.copy().astype(float)
    max_shift = math.floor(d + 1) # maximum hours we can spill over to.
    for i in range(1, max_shift + 1):
        if i < max_shift:
            new_v[i:] += v[:-i] # this hour is fully occupied
        else:
            new_v[i:] += (d - math.floor(d)) * v[:-i] # this hour only gets part of the visits.
    return new_v

def clip_poi_attributes_in_msa_df(d, min_datetime, max_datetime,
                                  clip_areas, clip_dwell_times, clip_visits,
                                  area_below=AREA_CLIPPING_BELOW,
                                  area_above=AREA_CLIPPING_ABOVE,
                                  dwell_time_above=DWELL_TIME_CLIPPING_ABOVE,
                                  visits_above=HOURLY_VISITS_CLIPPING_ABOVE,
                                  subcat_cutoff=SUBCATEGORY_CLIPPING_THRESH,
                                  topcat_cutoff=TOPCATEGORY_CLIPPING_THRESH):
    '''
    Deal with POI outliers by clipping their hourly visits, dwell times, and physical areas
    to some percentile of the corresponding distribution for each POI category.
    '''
    attr_cols = []
    if clip_areas:
        attr_cols.append('safegraph_computed_area_in_square_feet')
    if clip_dwell_times:
        attr_cols.append('avg_median_dwell')
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    if clip_visits:
        attr_cols.extend(hour_cols)
    assert all([col in d.columns for col in attr_cols])
    print('Clipping areas: %s (below=%d, above=%d), clipping dwell times: %s (above=%d), clipping visits: %s (above=%d)' %
          (clip_areas, area_below, area_above, clip_dwell_times, dwell_time_above, clip_visits, visits_above))

    subcats = []
    left_out_subcats = []
    indices_covered = []
    subcategory2idx = d.groupby('sub_category').indices
    for cat, idx in subcategory2idx.items():
        if len(idx) >= subcat_cutoff:
            subcats.append(cat)
            indices_covered.extend(idx)
        else:
            left_out_subcats.append(cat)
    num_subcat_pois = len(indices_covered)

    # group by top_category for POIs whose sub_category's are too small
    topcats = []
    topcategory2idx = d.groupby('top_category').indices
    remaining_pois = d[d['sub_category'].isin(left_out_subcats)]
    necessary_topcats = set(remaining_pois.top_category.unique())  # only necessary to process top_category's that have at least one remaining POI
    for cat, idx in topcategory2idx.items():
        if cat in necessary_topcats and len(idx) >= topcat_cutoff:
            topcats.append(cat)
            new_idx = np.array(list(set(idx) - set(indices_covered)))  # POIs that are not covered by sub_category clipping
            assert len(new_idx) > 0
            topcategory2idx[cat] = (idx, new_idx)
            indices_covered.extend(new_idx)

    print('Found %d sub-categories with >= %d POIs and %d top categories with >= %d POIs -> covers %d POIs' %
          (len(subcats), subcat_cutoff, len(topcats), topcat_cutoff, len(indices_covered)))
    kept_visits = np.nansum(d.iloc[indices_covered][hour_cols].values)
    all_visits = np.nansum(d[hour_cols].values)
    lost_visits = all_visits - kept_visits
    lost_pois = len(d) - len(indices_covered)
    print('Could not cover %d/%d POIs (%.1f%% POIs, %.1f%% hourly visits) -> dropping these POIs' %
          (lost_pois, len(d), 100. * lost_pois/len(d), 100 * lost_visits / all_visits))
    
    if lost_pois / len(d) > .2:#.03:
        raise Exception(f'Dropping too many POIs ({lost_pois / len(d)}% )during clipping phase')

    all_cats = topcats + subcats  # process top categories first so sub categories will compute percentiles on raw data
    new_data = np.array(d[attr_cols].copy().values)  # n_pois x n_cols_to_clip
    thresholds = np.zeros((len(all_cats), len(attr_cols)+1))  # clipping thresholds for category x attribute
    medians = np.zeros((len(all_cats), len(attr_cols)))  # medians for category x attribute
    indices_processed = []
    for i, cat in enumerate(all_cats):
        if i < len(topcats):
            cat_idx, new_idx = topcategory2idx[cat]
        else:
            cat_idx = subcategory2idx[cat]
            new_idx = cat_idx
        indices_processed.extend(new_idx)
        first_col_idx = 0  # index of first column for this attribute

        if clip_areas:
            cat_areas = new_data[cat_idx, first_col_idx]  # compute percentiles on entire category
            min_area = np.nanpercentile(cat_areas, area_below)
            max_area = np.nanpercentile(cat_areas, area_above)
            median_area = np.nanmedian(cat_areas)
            thresholds[i][first_col_idx] = min_area
            thresholds[i][first_col_idx+1] = max_area
            medians[i][first_col_idx] = median_area
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], min_area, max_area)
            first_col_idx += 1

        if clip_dwell_times:
            cat_dwell_times = new_data[cat_idx, first_col_idx]
            max_dwell_time = np.nanpercentile(cat_dwell_times, dwell_time_above)
            median_dwell_time = np.nanmedian(cat_dwell_times)
            thresholds[i][first_col_idx+1] = max_dwell_time
            medians[i][first_col_idx] = median_dwell_time
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], None, max_dwell_time)
            first_col_idx += 1

        if clip_visits:
            col_idx = np.arange(first_col_idx, first_col_idx+len(hour_cols))
            assert col_idx[-1] == (len(attr_cols)-1)
            orig_visits = new_data[cat_idx][:, col_idx].copy()  # need to copy bc will modify
            orig_visits[orig_visits == 0] = np.nan  # want percentile over positive visits
            # can't take percentile of col if it is all 0's or all nan's
            cols_to_process = col_idx[np.sum(~np.isnan(orig_visits), axis=0) > 0]
            max_visits_per_hour = np.nanpercentile(orig_visits[:, cols_to_process-first_col_idx], visits_above, axis=0)
            assert np.sum(np.isnan(max_visits_per_hour)) == 0
            thresholds[i][cols_to_process + 1] = max_visits_per_hour
            medians[i][cols_to_process] = np.nanmedian(orig_visits[:, cols_to_process-first_col_idx], axis=0)

            orig_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            orig_attributes = new_data[new_idx]  # return to un-modified version
            orig_attributes[:, cols_to_process] = np.clip(orig_attributes[:, cols_to_process], None, max_visits_per_hour)
            new_data[new_idx] = orig_attributes
            new_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            print('%s -> has %d POIs, processed %d POIs, %d visits before clipping, %d visits after clipping' %
              (cat, len(cat_idx), len(new_idx), orig_visit_sum, new_visit_sum))
        else:
            print('%s -> has %d POIs, processed %d POIs' % (cat, len(cat_idx), len(new_idx)))

    assert len(indices_processed) == len(set(indices_processed))  # double check that we only processed each POI once
    assert set(indices_processed) == set(indices_covered)  # double check that we processed the POIs we expected to process
    new_d = d.iloc[indices_covered].copy()
    new_d[attr_cols] = new_data[indices_covered]
    return new_d, all_cats, attr_cols, thresholds, medians

def apply_interventions_to_poi_cbg_matrices(
    poi_cbg_visits_list, poi_categories, poi_areas,
    new_all_hours, intervention_hour_idx,
    alpha, extra_weeks_to_simulate,
    top_category=None, sub_category=None,
    interpolate=True):
    '''
    Simulates hypothetical mobility patterns by editing visit matrices.
    '''
    # find POIs of interest
    if top_category is not None: 
        top_category_poi_idx = (poi_categories['top_category'] == top_category).values
    else: #default
        top_category = 'any'
        top_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all
    
    if sub_category is not None: #default 
        sub_category_poi_idx = (poi_categories['sub_category'] == sub_category).values
    else:
        sub_category = 'any'
        sub_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all
    
    # POI indices to intervene on
    intervened_poi_idx = top_category_poi_idx & sub_category_poi_idx  
    assert intervened_poi_idx.sum() > 0
    print("Intervening on POIs with top_category=%s, sub_category=%s (n=%i)" % (
        top_category, sub_category, intervened_poi_idx.sum()))

    # extend matrix list to extra weeks, loop final week for now
    num_pois, num_cbgs = poi_cbg_visits_list[0].shape
    
    # matrices to use for mobility:
    # all the observed range
    new_matrix_list = [m.copy() for m in poi_cbg_visits_list]
    
    # and new matrices taken at the end of the period
    for i in range(extra_weeks_to_simulate * 168):
        # get corresponding matrix from final week
        matrix_idx = -168 + (i % 168)  # always negative
        new_matrix_list.append(poi_cbg_visits_list[matrix_idx].copy())
        assert new_matrix_list[-1].shape == (num_pois, num_cbgs), len(new_matrix_list)-1
    assert len(new_matrix_list) == len(new_all_hours)

    # apply intervention to all POIs
    if top_category == 'any' and sub_category == 'any':  
        full_activity_sum = 0
        simulated_activity_sum = 0
        for i in range(intervention_hour_idx, len(new_all_hours)):
            no_reopening = new_matrix_list[i]
            full_reopening = new_matrix_list[i % 168]
            full_activity_sum += full_reopening.sum()
            if alpha == 1:
                new_matrix_list[i] = full_reopening.copy()
                simulated_activity_sum = full_activity_sum
            else:
                if interpolate: #default
                    new_matrix_list[i] = full_reopening.multiply(alpha) + no_reopening.multiply(1-alpha)
                else:
                    new_matrix_list[i] = full_reopening.multiply(alpha)
                simulated_activity_sum += new_matrix_list[i].sum()
        diff = full_activity_sum - simulated_activity_sum
        overall_cost = (100. * diff / full_activity_sum)
        print('Overall Cost (%% of full activity): %2.3f%%' % overall_cost)
        return new_matrix_list, {'overall_cost':overall_cost, 'cost_within_intervened_pois':overall_cost}

    # full activity based on first week of visits
    range_end = max(intervention_hour_idx + 168, len(poi_cbg_visits_list))
    
    # get corresponding matrix in first week
    full_activity = [poi_cbg_visits_list[i % 168] for i in range(intervention_hour_idx, range_end)]  
    full_activity = hstack(full_activity, format='csr')
    
    orig_activity = new_matrix_list[intervention_hour_idx:range_end]
    orig_activity = hstack(orig_activity, format='csr')
    assert full_activity.shape == orig_activity.shape
    print('Computed hstacks of sparse matrices [shape=(%d, %d)]' % full_activity.shape)

    # take mixture of full activity and original activity for POIs of interest
    indicator_vec = np.zeros(num_pois)
    indicator_vec[intervened_poi_idx] = 1.0
    alpha_vec = alpha * indicator_vec
    scaled_full_activity = full_activity.transpose().multiply(alpha_vec).transpose()
    if interpolate: #default
        non_alpha_vec = 1.0 - alpha_vec   # intervened POIs will have alpha*full + (1-alpha)*closed
    else:
        non_alpha_vec = 1.0 - indicator_vec  # intervened POIs will have alpha*full
    scaled_orig_activity = orig_activity.transpose().multiply(non_alpha_vec).transpose()
    activity_mixture = scaled_full_activity + scaled_orig_activity
    print('Computed mixture of full and original activity')

    # compute costs
    full_overall_sum = full_activity.sum()
    mixture_overall_sum = activity_mixture.sum()
    overall_diff = full_overall_sum - mixture_overall_sum
    overall_cost = (100. * overall_diff / full_overall_sum)
    print('Overall Cost (%% of full activity): %2.3f%%' % overall_cost)
    
    full_intervened_sum = full_activity.transpose().multiply(indicator_vec).sum()
    mixture_intervened_sum = activity_mixture.transpose().multiply(indicator_vec).sum()
    intervened_diff = full_intervened_sum - mixture_intervened_sum
    cost_within_intervened_pois = (100. * intervened_diff / full_intervened_sum)
    print('Cost within intervened POIs: %2.3f%%' % cost_within_intervened_pois)

    print('Redistributing stacked matrix into hourly pieces...')
    ts = time.time()
    looping = False
    for i in range(intervention_hour_idx, len(new_all_hours)):
        matrix_idx = i - intervention_hour_idx
        if i >= len(poi_cbg_visits_list) and matrix_idx >= 168:
            # once we are operating past the length of real data, the "original" matrix
            # is just the matrix from the last week of the real data for the corresponding
            # day, and if matrix_idx > 168, then the mixture for that corresponding day
            # has been computed already
            new_matrix_list[i] = new_matrix_list[i - 168].copy()
            if looping is False:
                print('Entering looping phase at matrix %d!' % matrix_idx)
                looping = True
        else:
            matrix_start = matrix_idx * num_cbgs
            matrix_end = matrix_start + num_cbgs
            new_matrix_list[i] = activity_mixture[:, matrix_start:matrix_end]
        assert new_matrix_list[i].shape == (num_pois, num_cbgs), 'intervention idx = %d, overall idx = %d [found size = (%d, %d)]' % (matrix_idx, i, new_matrix_list[i].shape[0], new_matrix_list[i].shape[1])
        if matrix_idx % 24 == 0:
            te = time.time()
            print('Finished matrix %d: time so far per hourly matrix = %.2fs' % (matrix_idx, (te-ts)/(matrix_idx+1)))
    return new_matrix_list, {'overall_cost':overall_cost, 'cost_within_intervened_pois':cost_within_intervened_pois}

def apply_interventions_to_poi_cbg_matrices_simplified(
    poi_cbg_visits_list, poi_categories, intervention_hour_idx, alpha, 
    extra_weeks_to_simulate=0, top_category=None, sub_category=None):
    '''
    Simulates hypothetical mobility patterns by editing visit matrices.
    '''
    # 

    # find POIs of interest
    # if top_category is not None: 
    #     top_category_poi_idx = (poi_categories['top_category'] == top_category).values
    # else: #default
    #     top_category = 'any'
    #     top_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all

    # if sub_category is not None: #default 
    #     sub_category_poi_idx = (poi_categories['sub_category'] == sub_category).values
    # else:
    #     sub_category = 'any'
    #     sub_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all

    cat_dict = {
        'retail': [470],                # Commercio al dettaglio, escluso quello di autoveicoli e di motocicli (470)
        'transport': [490, 500, 510],   # Trasporto: terrestre e mediante condotte (490) + 
                                        # marittimi e per vie d'acqua (500) + aereo (510)
        'accommodation': [550],         # Servizi di alloggio (550)
        'restaurant': [560],            # Attivita di servizi di ristorazione (560)
        'welfare': [860, 870, 880],     # Attivita dei servizi sanitari (860) + Servizi di assistenza residenziale (870) 
                                        # + non residenziale (880) 
    }

    if top_category is not None: 
        if not (top_category in ['no_retail','other_sectors', 'no_retail_no_welfare']):
            top_category_poi_idx = (poi_categories['top_category'].isin(cat_dict[top_category])).values
        elif top_category == 'no_retail': 
            top_category_poi_idx = (poi_categories['top_category'] != 470).values
        elif top_category == 'no_retail_no_welfare':  
            top_category_poi_idx = ((poi_categories['top_category'].isin([470,860, 870, 880])) == False).values                      
        elif top_category == 'other_sectors':
            top_category_poi_idx = (
                (poi_categories['top_category'].isin([ll for l in cat_dict.values() for ll in l])) == False).values
    else: #default
        top_category = 'any'
        top_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all

    if sub_category is not None: #default 
        if not (sub_category in ['no_retail','other_sectors', 'no_retail_no_welfare']):
            sub_category_poi_idx = (poi_categories['sub_category'].isin(cat_dict[sub_category])).values
        elif sub_category == 'no_retail': 
            sub_category_poi_idx = (poi_categories['sub_category'] != 470).values
        elif sub_category == 'no_retail_no_welfare':  
            sub_category_poi_idx = ((poi_categories['sub_category'].isin([470,860, 870, 880])) == False).values                      
        elif sub_category == 'other_sectors':
            sub_category_poi_idx = (
                (poi_categories['sub_category'].isin([ll for l in cat_dict.values() for ll in l])) == False).values
    else:
        sub_category = 'any'
        sub_category_poi_idx = np.ones(len(poi_categories)).astype(bool) # any == all    

    # POI indices to intervene on
    intervened_poi_idx = top_category_poi_idx & sub_category_poi_idx  
    assert intervened_poi_idx.sum() > 0
    print("Intervening on POIs with top_category=%s, sub_category=%s (n=%i)" % (
        top_category, sub_category, intervened_poi_idx.sum()))
    
    num_pois, _ = poi_cbg_visits_list[0].shape

    # take mixture of full activity and original activity for POIs of interest without extra weeks
    assert extra_weeks_to_simulate == 0    
    indicator_vec = np.zeros(num_pois)
    indicator_vec[intervened_poi_idx] = 1
    alpha_vec = (alpha * indicator_vec)+np.ones(num_pois)
    
    print(set(alpha_vec))
    
    # compute costs
    full_activity = [m.copy() for m in poi_cbg_visits_list]
    activity_mixture = full_activity[:intervention_hour_idx]
    scaled_full_activity = [
        m.copy().transpose().multiply(alpha_vec).transpose() 
        for m in poi_cbg_visits_list[intervention_hour_idx:]]
    activity_mixture.extend(scaled_full_activity)
    assert len(full_activity) == len(activity_mixture)

    full_overall_sum = sum([
        m.sum() for m in full_activity[intervention_hour_idx:]])
    mixture_overall_sum = sum([
        m.sum() for m in activity_mixture[intervention_hour_idx:]])
    overall_diff = full_overall_sum - mixture_overall_sum
    overall_cost = (100. * overall_diff / full_overall_sum)
    print('Overall Cost (%% of full activity): %2.3f%%' % overall_cost)

    full_intervened_sum = sum([
        m.transpose().multiply(indicator_vec).sum() 
        for m in full_activity[intervention_hour_idx:]]) 
    mixture_intervened_sum = sum([
        m.transpose().multiply(indicator_vec).sum() 
        for m in activity_mixture[intervention_hour_idx:]]) 
    intervened_diff = full_intervened_sum - mixture_intervened_sum
    cost_within_intervened_pois = (100. * intervened_diff / full_intervened_sum)
    print('Cost within intervened POIs: %2.3f%%' % cost_within_intervened_pois)
    costs = {'overall_cost':overall_cost, 'cost_within_intervened_pois':cost_within_intervened_pois}
    return activity_mixture, costs


def apply_distancing_degree(
    poi_cbg_visits_list, 
    distancing_degree, 
    msa_name,
    kind='original', 
    thresh_date=None, 
    offset_date=None, 
    delta_days=None):
    """
    ---------------------- Originally
    After the first week of March, assume that activity is an interpolation between true activity and 
    first-week-of-March activity.

    ---------------------- Current version.
    
    Given a date <thresh_date> for the start of the policy period, before that date the data is preserved and 
    after that date it is transformed to an interpolation between true activity and activity from the previous period
    (i.e. without restrictions). 

    <kind> specifies how activity from the past is selected: 
    - The "original" method selected data deterministically, extracting previous days of activty in ordered manner.
    - The "cf_list" method creates a sample of previous days and sample from it [TODO: separate by days]
    - The "bootstrap" method is still a working in progress and is currently not available

    Finally the <offset_date> parameters allows to skip a certain range of days and start select data from the past
    from the specified date.

    Note: at the moment this should work only for phase 2. Applying to phase 1 will probably fail. 

    TODO: method to select data from the previous year

    """    
    if thresh_date is None:
        if wave == 'first':
            THRESH_DATETIME = datetime.datetime(2020,3,9) # date of policies            
        elif wave == 'second':
            THRESH_DATETIME = datetime.datetime(2020,11,6) # date of policies
            #THRESH_DATETIME = datetime.datetime(2020, 10, 24)
        #THRESH_DATETIME = datetime.datetime(2020,10,23) # two weeks before date of policies
    else:
        THRESH_DATETIME = thresh_date
    
    if delta_days is None:
        delta = datetime.timedelta(days = 28) 
    else:
        delta = datetime.timedelta(days = delta_days)
    
    if offset_date is None:
        OFFSET_DATETIME = THRESH_DATETIME - delta
    else:
        OFFSET_DATETIME = offset_date        

    daterange_thresh = pd.date_range(
        start=MIN_DATETIME, 
        end=THRESH_DATETIME, 
        freq='D').strftime("%Y-%m-%d").tolist()
    thresh = len(daterange_thresh)*24
    
    daterange_offset = pd.date_range(
        start=MIN_DATETIME, 
        end=OFFSET_DATETIME, 
        freq='D').strftime("%Y-%m-%d").tolist()
    offset_thresh = len(daterange_offset)*24

    if kind == 'original':
        if (distancing_degree in [-1, -2]) == False: 
            return apply_distancing_degree_original(
                poi_cbg_visits_list, distancing_degree, 
                thresh=thresh, offset=offset_thresh)
        else:            
            if distancing_degree == -1:
                return apply_distancing_degree_previous_year(
                    msa_name = msa_name,
                    poi_cbg_visits_list = poi_cbg_visits_list, 
                    thresh=thresh)
            # NEW_rebuttal_20240106
            if distancing_degree == -2:
                return apply_distancing_degree_next_year(
                    msa_name = msa_name,
                    poi_cbg_visits_list = poi_cbg_visits_list, 
                    thresh=thresh)


    elif kind == 'cf_list':
        if distancing_degree == -1:
            return apply_distancing_degree_previous_year(
                msa_name = msa_name,
                poi_cbg_visits_list = poi_cbg_visits_list, 
                thresh=thresh)

        # all observations starting from 09-04 (end of summer for workers/students)
        # daily_ts = []        
        # for i in np.arange((daterange_thresh.index('2020-09-04')*24), thresh):
        #     daily_ts.append(poi_cbg_visits_list[i])             
        daily_ts = {i: [] for i in range(7)}
        for i, d in enumerate(pd.date_range(start=MIN_DATETIME, end=THRESH_DATETIME,freq='D')):
            if d >= OFFSET_DATETIME:
                for h in poi_cbg_visits_list[(i*24):(i+1)*24]:
                    daily_ts[d.dayofweek].append(h)  
        return apply_distancing_degree_counterfactual_list(
            poi_cbg_visits_list, distancing_degree, daily_ts, 
            thresh=thresh, thresh_weekday=THRESH_DATETIME.weekday())
    elif kind == 'bootstrap':
        print("Warning: bootstrap method is not ready yet. Exiting.")
        return


def apply_distancing_degree_original(
    poi_cbg_visits_list, distancing_degree, thresh=168, offset=0 ):
    """
    original threshold = 168 (first week)
    """
    new_visits_list = []
    weeklist_reduced = poi_cbg_visits_list[offset : thresh]
    for i, m in enumerate(poi_cbg_visits_list):
        if i < thresh:  # first week
            new_visits_list.append(m.copy())
        else:            
            first_week_m = weeklist_reduced[(i-offset) % (thresh-offset)]
            mixture = first_week_m.multiply(1-distancing_degree) + m.multiply(distancing_degree)
            new_visits_list.append(mixture.copy())
    return new_visits_list

def apply_distancing_degree_counterfactual_list(
    poi_cbg_visits_list, distancing_degree, cf_list, thresh=2064, thresh_weekday=6):
    """
    Sample randomly from previous days, respecting day of the week.
    """
    days = [0,1,2,3,4,5,6]
    new_visits_list = []
    for i, m in enumerate(poi_cbg_visits_list):
        if i < thresh:  # first week
            new_visits_list.append(m.copy())
        else:
            # get only data from a specific day of the week (avoid problems with outliers)
            days_sample = cf_list[thresh_weekday]
            # select one hourly matrix
            cf_matrix = days_sample[np.random.choice(len(days_sample))]
            # mix
            mixture = cf_matrix.multiply(1-distancing_degree) + m.multiply(distancing_degree)
            new_visits_list.append(mixture.copy())            
            # get next day
            if (i % 24 == 0):
                thresh_weekday = (thresh_weekday + 1) % len(days)
    return new_visits_list

def apply_shift_in_days(
    poi_cbg_visits_list, shift_in_days, kind='cf_list',     
    thresh_date=None, 
    offset_date=None, 
    delta_days=None):
    """
    Shift entire visits timeline shift_in_days days forward or backward,
    filling in the beginning or end as necessary with data from the first or last week.
    """

    if thresh_date is None:
        if wave == 'first':
            THRESH_DATETIME = datetime.datetime(2020,3,9) # date of policies
        elif wave == 'second':
            #THRESH_DATETIME = datetime.datetime(2020,10,23) # two weeks before date of policies
            THRESH_DATETIME = datetime.datetime(2020,11,6) # date of policies
            #THRESH_DATETIME = datetime.datetime(2020, 10, 24)            
    else:
        THRESH_DATETIME = thresh_date
    
    if delta_days is None:
        delta = datetime.timedelta(days = 28) 
    else:
        delta = datetime.timedelta(days = delta_days)
    
    if offset_date is None:
        if shift_in_days > 0:
            OFFSET_DATETIME = THRESH_DATETIME - delta
        else:
            OFFSET_DATETIME = THRESH_DATETIME + delta
    else:
        OFFSET_DATETIME = offset_date      

    daterange_thresh = pd.date_range(
        start=MIN_DATETIME, 
        end=THRESH_DATETIME, 
        freq='D').strftime("%Y-%m-%d").tolist()
    thresh = len(daterange_thresh)*24

    if kind == 'original':
        return apply_shift_in_days_original(
            poi_cbg_visits_list, shift_in_days)
    elif kind == 'cf_list':        
        daily_ts = {i: [] for i in range(7)}
        if shift_in_days > 0:
            for i, d in enumerate(pd.date_range(start=MIN_DATETIME, end=THRESH_DATETIME,freq='D')):
                if d >= OFFSET_DATETIME:
                    for h in poi_cbg_visits_list[(i*24):(i+1)*24]:
                        daily_ts[d.dayofweek].append(h)  
        else:
            for i, d in enumerate(pd.date_range(start=MIN_DATETIME, end=OFFSET_DATETIME,freq='D')):
                if d >= THRESH_DATETIME:
                    for h in poi_cbg_visits_list[(i*24):(i+1)*24]:
                        daily_ts[d.dayofweek].append(h)  

        return apply_shift_in_days_counterfactual_list(
            poi_cbg_visits_list, shift_in_days, daily_ts, thresh_weekday=THRESH_DATETIME.weekday())

def apply_shift_in_days_counterfactual_list(
    poi_cbg_visits_list, shift_in_days, cf_list, thresh_weekday):
    """
    Shift entire visits timeline shift_in_days days forward or backward,
    filling in the beginning or end as necessary with data from the first or last week.
    """
    new_visits_list = []
    shift_in_hours = shift_in_days * 24
    days = [0,1,2,3,4,5,6]             
    
    if shift_in_hours <= 0:  # shift earlier
        new_visits_list = [m.copy() for m in poi_cbg_visits_list[abs(shift_in_hours):]]
        current_length = len(new_visits_list)
        for i in range(current_length, len(poi_cbg_visits_list)):                 
            days_sample = cf_list[thresh_weekday]
            last_week_counterpart = days_sample[np.random.choice(len(days_sample))].copy()
            new_visits_list.append(last_week_counterpart)
            if (i % 24 == 0):
                thresh_weekday = (thresh_weekday + 1) % len(days)
    else:  # shift later
        for i in range(len(poi_cbg_visits_list)):
            if (i - shift_in_hours) < 0:
                # fill in with hourly data sampled randomly
                days_sample = cf_list[thresh_weekday]
                first_week_counterpart = days_sample[np.random.choice(len(days_sample))].copy()
                new_visits_list.append(first_week_counterpart)
                if (i % 24 == 0):
                    thresh_weekday = (thresh_weekday + 1) % len(days)
            else:
                new_visits_list.append(poi_cbg_visits_list[i].copy())
    assert len(new_visits_list) == len(poi_cbg_visits_list)
    return new_visits_list

def apply_shift_in_days_original(poi_cbg_visits_list, shift_in_days):
    """
    Shift entire visits timeline shift_in_days days forward or backward,
    filling in the beginning or end as necessary with data from the first or last week.
    """
    new_visits_list = []
    shift_in_hours = shift_in_days * 24
    if shift_in_hours <= 0:  # shift earlier
        new_visits_list = [m.copy() for m in poi_cbg_visits_list[abs(shift_in_hours):]]
        current_length = len(new_visits_list)
        assert current_length >= 168
        last_week = new_visits_list[-168:]
        for i in range(current_length, len(poi_cbg_visits_list)):
            last_week_counterpart = last_week[i % 168].copy()
            new_visits_list.append(last_week_counterpart)
    else:  # shift later
        for i in range(len(poi_cbg_visits_list)):
            if i-shift_in_hours < 0:
                # fill in with the last part of the first week.
                # so eg if shift_in_hours is 72, we take the last 72 hours of the first week.
                first_week_idx = (168 - shift_in_hours + i) % 168

                # alternate, more complex computation as sanity check.
                distance_from_start = (shift_in_hours - i) % 168
                first_week_idx_2 = (168 - distance_from_start) % 168

                assert first_week_idx_2 == first_week_idx
                new_visits_list.append(poi_cbg_visits_list[first_week_idx].copy())
            else:
                new_visits_list.append(poi_cbg_visits_list[i-shift_in_hours].copy())
    assert len(new_visits_list) == len(poi_cbg_visits_list)
    return new_visits_list


def apply_distancing_degree_previous_year(msa_name, poi_cbg_visits_list, thresh):
    # HARDCODING
    visits_list_2019, visits_list_2020 = align_mobility_data_from_different_years(
        msa_name, poi_cbg_visits_list, 2019, 2020)    
    new_visits_list = []
    for i, m in enumerate(visits_list_2020):
        if i < thresh:  # first week
            new_visits_list.append(m.copy())
        else:            
            new_visits_list.append(visits_list_2019[i])
    return new_visits_list

# NEW_rebuttal_20240106
def apply_distancing_degree_next_year(msa_name, poi_cbg_visits_list, thresh):
    # HARDCODING
    visits_list_2021, visits_list_2020 = align_mobility_data_from_different_years(
        msa_name, poi_cbg_visits_list, 2021, 2020)    
    new_visits_list = []
    for i, m in enumerate(visits_list_2020):
        if i < thresh:  # first week
            new_visits_list.append(m.copy())
        else:            
            new_visits_list.append(visits_list_2021[i])
    return new_visits_list

def apply_category_grid(msa_name, poi_cbg_visits_list, target_cats):
    # HARDCODING
    files_year_1, files_year_2 = align_mobility_data_from_different_years(
        msa_name, poi_cbg_visits_list, 2019, 2020, return_all_files=True)
    poi_cbg_visits_list_2019_new, new_metadata_2019_pairs, _ = files_year_1
    poi_cbg_visits_list_2020_new, _ , _ = files_year_2    

    step_dict = {cat:steps*CATEGORY_STEPS[cat] for cat, steps in target_cats}
    # move all these three sectors together
    step_dict['870'] = step_dict['860']
    step_dict['880'] = step_dict['860']

    filter_cat = []
    for pair in new_metadata_2019_pairs:
        if pair[1] in step_dict.keys():
            filter_cat.append(step_dict[pair[1]])
        else:
            filter_cat.append(0)
    
    # check that filter is empty only when all sectors are not shifted
    assert (sum(filter_cat) > 0) | (sum(step_dict.values()) == 0) 

    new_visits_list = [
        m20 + m19.multiply(np.array(filter_cat)[:, np.newaxis])
        for m19, m20 in zip(poi_cbg_visits_list_2019_new, poi_cbg_visits_list_2020_new)]       
    
    # HARDCODING
    MIN_DATETIME = datetime.datetime(2020, 10, 9, 0)
    TRAIN_TEST_PARTITION = datetime.datetime(2020, 11, 6)

    # MIN_DATETIME = datetime.datetime(2020, 9, 10, 0)
    # TRAIN_TEST_PARTITION = datetime.datetime(2020, 10, 24)    

    thresh = len(pd.date_range(
        start=MIN_DATETIME, 
        end=TRAIN_TEST_PARTITION, 
        freq='H'))

    final_visit_list = []
    for i in range(len(poi_cbg_visits_list_2020_new)):
        if i < thresh: 
            final_visit_list.append(poi_cbg_visits_list_2020_new[i])
        else:
            final_visit_list.append(new_visits_list[i])

    return final_visit_list

def align_mobility_data_from_different_years(msa_name, poi_cbg_visits_list_ref, year1, year2, return_all_files=False):

    poi_cbg_visits_list_2019, metadata_2019, population_2019 = get_year_details(
        year1, msa_name)
    poi_cbg_visits_list_2019 = [m.transpose() for m in poi_cbg_visits_list_2019]
    
    _, metadata_2020, population_2020 = get_year_details(
        year2, msa_name)

    a = set(population_2020['municipality']) 
    b = set(population_2019['municipality'])

    keep_pop_2020 = population_2020['municipality'].isin(a-b) == False
    keep_pop_2019 = population_2019['municipality'].isin(b-a) == False

    assert population_2020[keep_pop_2020][
        'municipality'].tolist() == population_2019[
            keep_pop_2019]['municipality'].tolist()

    metadata_2020_pairs = list(zip(
        metadata_2020[msa_name]['POI_municipalities'], 
        metadata_2020[msa_name]['POI_categories']))

    metadata_2019_pairs = list(zip(
        metadata_2019[msa_name]['POI_municipalities'], 
        metadata_2019[msa_name]['POI_categories']))

    col_to_drop_2020 = []
    keep_meta_2020 = []
    for p in metadata_2020_pairs:
        if p not in metadata_2019_pairs:
            col_to_drop_2020.append(p)
            keep_meta_2020.append(False)
        else:
            keep_meta_2020.append(True)
            
    col_to_drop_2019 = []
    keep_meta_2019 = []
    for p in metadata_2019_pairs:
        if p not in metadata_2020_pairs:
            col_to_drop_2019.append(p)
            keep_meta_2019.append(False)
        else:
            keep_meta_2019.append(True)

    new_metadata_2020_pairs = [
        p for p,t in zip(metadata_2020_pairs, keep_meta_2020) if t]
    new_metadata_2019_pairs = [
        p for p,t in zip(metadata_2019_pairs, keep_meta_2019) if t]

    assert new_metadata_2020_pairs == new_metadata_2019_pairs

    poi_cbg_visits_list_2019_new = [
        p[keep_meta_2019] for p in poi_cbg_visits_list_2019]
    poi_cbg_visits_list_2019_new = [
        p[:, keep_pop_2019] for p in poi_cbg_visits_list_2019_new]
    
    # repeat the first day to match 29th of February (sorry)
    if (wave == 1) or (wave=='first'):
        poi_cbg_visits_list_2019_new = poi_cbg_visits_list_2019_new[:24] + poi_cbg_visits_list_2019_new

    poi_cbg_visits_list_2020_new = [
        p[keep_meta_2020] for p in poi_cbg_visits_list_ref]
    poi_cbg_visits_list_2020_new = [
        p[:, keep_pop_2020] for p in poi_cbg_visits_list_2020_new]

    print(len(poi_cbg_visits_list_2020_new), len(poi_cbg_visits_list_2019_new))
    assert len(poi_cbg_visits_list_2020_new) == len(poi_cbg_visits_list_2019_new)
        
    assert np.all([
        i.shape == j.shape for i,j 
        in zip(poi_cbg_visits_list_2019_new,poi_cbg_visits_list_2020_new)]) 

    if return_all_files:
        population_2020_new = population_2020[keep_pop_2020].reset_index(drop=True)
        population_2019_new = population_2019[keep_pop_2019].reset_index(drop=True)
        files_year_1 = (poi_cbg_visits_list_2019_new, new_metadata_2019_pairs, population_2019_new)
        files_year_2 = (poi_cbg_visits_list_2020_new, new_metadata_2020_pairs, population_2020_new)
        return files_year_1, files_year_2
    else:   
        return poi_cbg_visits_list_2019_new,poi_cbg_visits_list_2020_new

def get_year_details(year, region, phase=wave, filter_to_use=data_type):
    paths = manage_exporting_paths(phase, filter_to_use, year=year)
    ipf_path,pop_path,meta_path = paths
    min_t, max_t = get_dates_by_phase(phase, year=year)

    ipf_filename = get_ipf_filename(
        region, min_t, max_t,
        True,True,output_path=ipf_path)
    assert os.path.exists(ipf_filename)
    with open(ipf_filename, 'rb') as mobility_path:
        poi_cbg_visits_list_y = pickle.load(mobility_path)

    metadata_filename = get_metadata_filename(    
        region, min_t, max_t,
        True,True,output_path=meta_path)
    assert os.path.exists(metadata_filename)
    with open(metadata_filename, 'rb') as meta_path:
        metadata_y = pickle.load(meta_path)

    population_filename = os.path.join(
        pop_path, f"{region}.csv")
    assert os.path.exists(population_filename)
    population_y = pd.read_csv(population_filename)
    return poi_cbg_visits_list_y, metadata_y, population_y

def apply_mobility_rescaling(
    poi_cbg_visits_list, 
    rescale_by, 
    thresh_date=None):
    if thresh_date is None:
        if wave == 'first':
            THRESH_DATETIME = datetime.datetime(2020,3,9) # date of policies
        elif wave == 'second':
            #THRESH_DATETIME = datetime.datetime(2020,10,23) # two weeks before date of policies
            THRESH_DATETIME = datetime.datetime(2020,11,6) # date of policies
            #THRESH_DATETIME = datetime.datetime(2020, 10, 24)                  
    else:
        THRESH_DATETIME = thresh_date
    
    daterange_thresh = pd.date_range(
        start=MIN_DATETIME, 
        end=THRESH_DATETIME, 
        freq='D').strftime("%Y-%m-%d").tolist()
    thresh = len(daterange_thresh)*24    
    
    new_visits_list = []
    for i, m in enumerate(poi_cbg_visits_list):
        if i < thresh:  # first week
            new_visits_list.append(m.copy())
        else:            
            new_visits_list.append(m.copy().multiply(rescale_by))
    return new_visits_list

# modified 4 March 2022	
# def sanity_check_error_metrics(fast_to_load_results):
#     """
#     Make sure train and test loss sum to total loss in the way we would expect.
#     """
#     n_train_days = len(helper.list_datetimes_in_range(
#         fast_to_load_results['train_loss_dict']['eval_start_time_cases'],
#         fast_to_load_results['train_loss_dict']['eval_end_time_cases']))

#     n_test_days = len(helper.list_datetimes_in_range(
#         fast_to_load_results['test_loss_dict']['eval_start_time_cases'],
#         fast_to_load_results['test_loss_dict']['eval_end_time_cases']))

#     n_total_days = len(helper.list_datetimes_in_range(
#         fast_to_load_results['loss_dict']['eval_start_time_cases'],
#         fast_to_load_results['loss_dict']['eval_end_time_cases']))

#     assert n_train_days + n_test_days == n_total_days
#     assert fast_to_load_results['loss_dict']['eval_end_time_cases'] == fast_to_load_results['test_loss_dict']['eval_end_time_cases']
#     assert fast_to_load_results['loss_dict']['eval_start_time_cases'] == fast_to_load_results['train_loss_dict']['eval_start_time_cases']
#     for key in ['daily_cases_MSE', 'cumulative_cases_MSE']:
#         if 'RMSE' in key:
#             train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] ** 2 +
#                  n_test_days * fast_to_load_results['test_loss_dict'][key] ** 2)

#             overall_loss = n_total_days * fast_to_load_results['loss_dict'][key] ** 2
#         else:
#             train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] +
#                  n_test_days * fast_to_load_results['test_loss_dict'][key])

#             overall_loss = n_total_days * fast_to_load_results['loss_dict'][key]

#         assert np.allclose(train_plus_test_loss, overall_loss, rtol=1e-6)
#     print("Sanity check error metrics passed")

def get_full_activity_num_visits(msa, intervention_datetime, extra_weeks_to_simulate, min_datetime, max_datetime):
    """
    Get the total number of visits post-intervention date assuming we just looped activity from the first week
    """
    fn = get_ipf_filename(msa, min_datetime, max_datetime, True, True)
    f = open(fn, 'rb')
    poi_cbg_visits_list = pickle.load(f)
    f.close()
    all_hours = helper.list_hours_in_range(min_datetime, max_datetime + datetime.timedelta(hours=168 * extra_weeks_to_simulate))
    assert(intervention_datetime in all_hours)
    intervention_hour_idx = all_hours.index(intervention_datetime)
    full_total = 0
    for t in range(intervention_hour_idx, len(all_hours)):
        full_activity_matrix = poi_cbg_visits_list[t % 168]
        full_total += full_activity_matrix.sum()
    return full_total, intervention_hour_idx

def get_lir_checkpoints_and_prop_visits_lost(timestring, intervention_hour_idx,
                                             full_activity_num_visits=None, group='all', normalize=True):
    """
    Returns the fraction of the population in state L+I+R at two checkpoints: at the point of reopening,
    and at the end of the simulation. Also returns the proportion of visits lost after the reopening,
    compared to full reopening.
    """
    model, kwargs, _, _, fast_to_load_results = load_model_and_data_from_timestring(timestring,
                                                                 load_fast_results_only=False,
                                                                 load_full_model=True)
    group_history = model.history[group]
    lir = group_history['latent'] + group_history['infected'] + group_history['removed']
    pop_size = group_history['total_pop']
    if normalize:
        intervention_lir = lir[:, intervention_hour_idx] / pop_size
        final_lir = lir[:, -1] / pop_size
    else:
        intervention_lir = lir[:, intervention_hour_idx]
        final_lir = lir[:, -1]
    intervention_cost = fast_to_load_results['intervention_cost']
    if 'total_activity_after_max_capacity_capping' in intervention_cost:
        # the max_capacity_capping and uniform reduction experiments save different activity measures
        # the max_capacity_capping experiments save 'total_activity_after_max_capacity_capping'
        # which needs to be translated into prop visits lost
        # the uniform reduction experiments save 'overall_cost' which is the percentage of visits lost
        # so it needs to be divided by 100 to be a decimal
        assert full_activity_num_visits is not None
        num_visits = intervention_cost['total_activity_after_max_capacity_capping']
        visits_lost = (full_activity_num_visits - num_visits) / full_activity_num_visits
    else:
        assert 'overall_cost' in intervention_cost
        visits_lost = intervention_cost['overall_cost'] / 100
    return intervention_lir, final_lir, visits_lost

def get_uniform_proportions_per_msa(min_timestring=None, max_cap_df=None, verbose=True):
    """
    Get the proportion of visits kept for each max capacity experiment, so that we can run the corresponding
    experiment with uniform reduction.
    """
    assert not(min_timestring is None and max_cap_df is None)
    if max_cap_df is None:
        max_cap_df = evaluate_all_fitted_models_for_experiment('test_max_capacity_clipping',
                                                       min_timestring=min_timestring)
    max_cap_df['MSA_name'] = max_cap_df['data_kwargs'].map(lambda x:x['MSA_name'])
    k = 'max_capacity_alpha'
    max_cap_df['counterfactual_%s' % k] = max_cap_df['counterfactual_poi_opening_experiment_kwargs'].map(lambda x:x[k])
    extra_weeks_to_simulate = max_cap_df.iloc[0]['counterfactual_poi_opening_experiment_kwargs']['extra_weeks_to_simulate']
    intervention_datetime = max_cap_df.iloc[0]['counterfactual_poi_opening_experiment_kwargs']['intervention_datetime']
    min_datetime = max_cap_df.iloc[0]['model_kwargs']['min_datetime']
    max_datetime = max_cap_df.iloc[0]['model_kwargs']['max_datetime']

    msa2proportions = {}
    for msa in max_cap_df.MSA_name.unique():
        full_activity, intervention_idx = get_full_activity_num_visits(msa,
                                               intervention_datetime=intervention_datetime,
                                               extra_weeks_to_simulate=extra_weeks_to_simulate,
                                               min_datetime=min_datetime,
                                               max_datetime=max_datetime)
        msa_df = max_cap_df[max_cap_df['MSA_name'] == msa]
        values = sorted(msa_df['counterfactual_max_capacity_alpha'].unique())
        proportions = []
        for v in values:
            first_ts = msa_df[msa_df.counterfactual_max_capacity_alpha == v].iloc[0].timestring
            _, _, visits_lost = get_lir_checkpoints_and_prop_visits_lost(first_ts,
                        intervention_idx, group='all', full_activity_num_visits=full_activity)
            proportions.append(np.round(1 - visits_lost, 5))
        msa2proportions[msa] = proportions
        if verbose:
            print(msa, proportions)
    return msa2proportions