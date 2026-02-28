from covid_constants_and_util import *
import argparse
import pickle
#from utilities import *
from run_one_model import fit_and_save_one_model
from run_parallel_models import generate_data_and_model_configs, partition_jobs_across_computers,run_many_models_in_parallel,print_config_as_json
#from model_evaluation import *

if __name__ == '__main__':    
    # command line arguments.
    # Basically, this script can be called two ways: 
    #     1. as a manager job: generates configs and fires off a bunch of 
    #        worker jobs
    #     2. as a worker job: runs a single model with a single config.
    # The command line argument "manager_or_worker_job" specifies which of 
    # these two usages we're using.
    #
    # The other important command line argument is "experiment_to_run", 
    # which specifies which step of the experimental pipeline we're running.
    #
    # The worker jobs take additional arguments like 
    # - timestring, which specifies the timestring we use to save model files
    # - config_idx, which specifies which config we're using.
    #
    # TODO: 
    #   - call this script main?
    
    # TODO: Move to config?
    # TODO: Use as options for the experiment_to_run argument?
    valid_experiments = [
        'just_save_ipf_output', 
        'normal_grid_search', 
        'grid_search_home_proportion_beta',
        'grid_search_aggregate_mobility',
        'calibrate_r0', 
        'calibrate_r0_aggregate_mobility',        
        'test_interventions',
        'test_retrospective_counterfactuals', 
        'test_retrospective_counterfactuals_rebuttal',
        'test_max_capacity_clipping',
        'test_uniform_proportion_of_full_reopening', 
        'rerun_best_models_and_save_cases_per_poi',
        'test_categories_on_grid'        
    ]
    parser = argparse.ArgumentParser()
    # TODO: transform to "single_model" switch
    parser.add_argument(
        'manager_or_worker_job', 
        help='Is this the manager job or the worker job?',
        choices=['run_many_models_in_parallel', 'fit_and_save_one_model'])
    parser.add_argument(
        'experiment_to_run', 
        help='The name of the experiment to run')
    parser.add_argument('--timestring', type=str)
    parser.add_argument('--config_idx', type=int)
    parser.add_argument(
        '--how_to_select_best_grid_search_models', 
        type=str, 
        choices=['daily_cases_rmse', 'daily_deaths_rmse', 'daily_cases_poisson'], 
        default='daily_cases_rmse')
    parser.add_argument(
        '--min_timestring_best_fit', 
        type=str, 
        help='Min timestring to use to locate normal_grid_search models',
        default='2020_07_16_10_4')
    parser.add_argument(
        '--max_timestring_best_fit', 
        type=str, 
        help='Max timestring to use to locate normal_grid_search models',
        default=None)        
    parser.add_argument(
        '--msa', 
        type=str, 
        help='Select msa from command line',
        default=None)        

    args = parser.parse_args()

    # Less frequently used arguments.
    config_idx_to_start_at = None
    skip_previously_fitted_kwargs = False
    
    # what grid search models do we look at when loading interventions.
    min_timestring = '2020_07_16_10_4'
    
    min_timestring_to_load_best_fit_models_from_grid_search = args.min_timestring_best_fit
    max_timestring_to_load_best_fit_models_from_grid_search = args.max_timestring_best_fit
    #min_timestring_to_load_best_fit_models_from_grid_search = '2020_07_16_10_4' 
    
    config_filename = '%s_configs.pkl' % COMPUTER_WE_ARE_RUNNING_ON.replace('.stanford.edu', '')

    if args.manager_or_worker_job == 'run_many_models_in_parallel':
        # manager job generates configs.
        
        # Argument checking
        assert args.timestring is None
        assert args.config_idx is None
        experiment_list = args.experiment_to_run.split(',')
        assert [a in valid_experiments for a in experiment_list] #TODO: move in argparse logic?
        
        print("Starting the following list of experiments")
        print(experiment_list)
        
        configs_to_fit = []
        for experiment in experiment_list:
            # Check arguments
            if experiment not in ['just_save_ipf_output', 'calibrate_r0', 'normal_grid_search','grid_search_aggregate_mobility']:
                assert args.how_to_select_best_grid_search_models is not None, 'Error: must specify how you wish to select best-fit models'
            
            # Create configuration
            configs_for_experiment = generate_data_and_model_configs(
                config_idx_to_start_at = config_idx_to_start_at,
                skip_previously_fitted_kwargs = skip_previously_fitted_kwargs,
                min_timestring = min_timestring,
                experiment_to_run = experiment,
                how_to_select_best_grid_search_models = args.how_to_select_best_grid_search_models,
                min_timestring_to_load_best_fit_models_from_grid_search = min_timestring_to_load_best_fit_models_from_grid_search, 
                max_timestring_to_load_best_fit_models_from_grid_search = max_timestring_to_load_best_fit_models_from_grid_search,
                provided_msa=args.msa, #NEW                
            )
            
            # Deal with paralleism
            configs_for_experiment = partition_jobs_across_computers(COMPUTER_WE_ARE_RUNNING_ON, configs_for_experiment)            
            configs_to_fit += configs_for_experiment
        
        # Dump all configurations to load them afterward using their IDs as reference
        print("Total number of configs to run on %s (%i experiments): %i" % (COMPUTER_WE_ARE_RUNNING_ON, len(configs_to_fit), len(experiment_list)))
        f = open(config_filename, 'wb')
        pickle.dump(configs_to_fit, f)
        f.close()
        
        # Fire off worker jobs.
        run_many_models_in_parallel(configs_to_fit)        

    elif args.manager_or_worker_job == 'fit_and_save_one_model':
        # worker job needs to load the list of configs and figure out which one it's running.
        assert args.experiment_to_run in valid_experiments        
        
        # Load all configurations saved in the previous step
        print("loading configs from %s" % config_filename)
        f = open(config_filename, 'rb')
        configs_to_fit = pickle.load(f)
        f.close()  
    
        # single worker job; use command line arguments to retrieve config and timestring.
        timestring = args.timestring
        config_idx = args.config_idx
        assert timestring is not None and config_idx is not None
        # Use experiment ID to load specific configuration
        data_and_model_config = configs_to_fit[config_idx]

        if 'grid_search' in args.experiment_to_run:
            train_test_partition = TRAIN_TEST_PARTITION
        else:
            train_test_partition = None

        print("Running single model. Kwargs are")
        print_config_as_json(data_and_model_config)

        fit_and_save_one_model(
            timestring,
            train_test_partition = train_test_partition,
            model_kwargs = data_and_model_config['model_kwargs'],
            data_kwargs = data_and_model_config['data_kwargs'],
            experiment_to_run = data_and_model_config['experiment_to_run'])
    else:
        raise Exception("This is not a valid way to call this method")
