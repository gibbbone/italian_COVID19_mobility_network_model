import argparse
from model_evaluation import filter_timestrings_for_properties
from utilities import get_dates_by_phase

def get_experiments_results(
    phase, region, min_t, experiment='normal_grid_search', max_t=None, 
    min_datetime=None, max_datetime=None):
    # Kept general to update it later
    required_properties = {'experiment_to_run':experiment}
    if (min_datetime is None) or (max_datetime is None):
        MIN_DATETIME,MAX_DATETIME = get_dates_by_phase(phase)
    else:
        MIN_DATETIME,MAX_DATETIME = min_datetime, max_datetime

    required_model_kwargs = {
        'min_datetime': MIN_DATETIME,
        'max_datetime': MAX_DATETIME}
    required_data_kwargs = {'MSA_name': region, 'nrows':None}

    res = filter_timestrings_for_properties(
        min_timestring= min_t,
        max_timestring= max_t,
        required_properties = required_properties,
        required_model_kwargs = required_model_kwargs,
        required_data_kwargs= required_data_kwargs
    )
    res.sort()

    print(f"Found {len(res)} models with the given properties:")
    print(f"> Experiment: {experiment}")
    print(f"> Region: {region}")
    print(f"> Min_timestring: {min_t}")
    print("----------------")
    if res:
        print(f"First model: {res[0]}")
        print(f"Last model: {res[-1]}")
    return res


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        '--min_t', 
        help='Find model output obtained after this date')
    parser.add_argument(
        '--max_t', 
        help='Find model output obtained before this date', 
        default=None)
    parser.add_argument(
        '--experiment', 
        help='The name of the experiment to run', 
        default='normal_grid_search')
    parser.add_argument(
        '--region', 
        help='Search for a specific region', 
        type=str)
    parser.add_argument(
        '--phase', 
        help='Specify relevant time period', 
        type=int, 
        required=True)
    args = parser.parse_args()
    
    res = get_experiments_results(
        experiment=args.experiment,
        phase=args.phase,
        region=args.region,
        min_t=args.min_t,
        max_t=args.max_t)