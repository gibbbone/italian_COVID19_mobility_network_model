import pickle
import time 
import gzip
from covid_constants_and_util import *
from utilities import *
import helper_methods_for_aggregate_data_analysis as helper
from model_evaluation import *

#########################################################
# Functions to evaluate model fit and basic results
#########################################################

def compare_model_vs_real_num_cases_subset(
    nyt_outcomes,
    mdl_start_time,
    subset='nyt',
    compare_start_time=None,
    compare_end_time=None,
    model=None,
    model_results=None,
    mdl_prediction=None,
    projected_hrs=None,
    detection_rate=.10,
    detection_lag=DETECTION_LAG, #previously was 17 in NEW_V2
    death_rate=.0066,
    death_lag=18,
    prediction_mode='deterministic',
    verbose=False,
    make_plot=False,
    ax=None,
    title=None,
    plot_log=False,
    plot_mode='cases',
    plot_errorbars=False,
    plot_real_data=True,
    plot_daily_not_cumulative=False,
    only_plot_intersection=True,
    model_line_label=None,
    true_line_label=None,
    x_interval=None,
    add_smoothed_real_data_line=True,
    title_fontsize=20,
    legend_fontsize=16,
    tick_label_fontsize=16,
    marker_size=5,
    plot_legend=True,
    real_data_color='black',
    model_color='darkorchid',
    xticks=None,
    x_range=None,
    y_range=None,
    only_two_yticks=False,
    return_mdl_pred_and_hours=False):
    
    assert plot_daily_not_cumulative in [True, False]
    assert prediction_mode in {'deterministic', 'exponential', 'gamma', 'model_history'}
    
    if model is not None:
        cbgs_to_idxs = model.CBGS_TO_IDXS
        history = model.history
        assert(subset in history)
        assert model_results is None
        assert mdl_prediction is None
        assert projected_hrs is None
    elif model_results is not None: #default
        # NEW_V2
        #cbgs_to_idxs = model_results['CBGS_TO_IDXS']
        history = model_results['history']
        assert(subset in history)
        assert mdl_prediction is None
        assert projected_hrs is None
    else:
        assert mdl_prediction is not None
        assert projected_hrs is not None

    # NEW_V2
    #real_dates, real_cases, real_deaths = get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes)
    real_dates, real_cases = get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes)
    score_dict = {}

    if mdl_prediction is not None:
        mdl_prediction_provided = True
    else: #default
        mdl_prediction_provided = False

    if not mdl_prediction_provided: #default
        # align cases with datetimes
        # should think of this as a cumulative count because once you enter the 
        # removed state, you never leave. So mdl_cases is the number of people who
        # have _ever_ been infectious or removed (ie, in states I or R).
        mdl_IR = (history[subset]['infected'] + history[subset]['removed']) 
        num_hours = mdl_IR.shape[1]
        mdl_end_time = mdl_start_time + datetime.timedelta(hours=num_hours-1)
               
        mdl_hours = helper.list_hours_in_range(mdl_start_time, mdl_end_time)
        mdl_dates = helper.list_datetimes_in_range(mdl_start_time, mdl_end_time)
        assert(mdl_start_time < mdl_end_time)
    else:
        mdl_IR = None

    #modes = ['cases', 'deaths']
    modes = ['cases']
    #print("Warning: data validation done only on cases, because we do not have data on deaths at the province level in Italian data.")
    for mode in modes:
        if mode == 'cases':
            real_data = real_cases
        else:
            #real_data = real_deaths
            print("Wrong turn, go back to start.")
            return
        if not mdl_prediction_provided:
            # note: mdl_prediction should always represent an hourly *cumulative* count per seed x hour
            if mode == 'cases':
                min_thresholds = [1, 10, 20, 50, 100]  # don't evaluate LL on very small numbers -- too noisy
                if prediction_mode == 'deterministic':  # assume constant detection rate and delay
                    mdl_prediction = mdl_IR * detection_rate
                    projected_hrs = [hr + datetime.timedelta(days=detection_lag) for hr in mdl_hours]
                elif prediction_mode == 'exponential':  # draw delays from exponential distribution
                    mdl_hourly_new_cases, _ = draw_cases_and_deaths_from_exponential_distribution(
                        mdl_IR,detection_rate, detection_lag, death_rate, death_lag)
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
                elif prediction_mode == 'gamma':  # draw delays from gamma distribution
                    mdl_hourly_new_cases, _ = draw_cases_and_deaths_from_gamma_distribution(
                        mdl_IR,detection_rate, death_rate)
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
                else:  # modeled confirmed cases during simulation
                    assert 'new_confirmed_cases' in history[subset]
                    mdl_hourly_new_cases = history[subset]['new_confirmed_cases']
                    mdl_prediction = get_cumulative(mdl_hourly_new_cases)
                    projected_hrs = mdl_hours
            else:
                min_thresholds = [1, 2, 3, 5, 10]  # don't evaluate LL on very small numbers -- too noisy
                if prediction_mode == 'deterministic':  # assume constant detection rate and delay
                    mdl_prediction = mdl_IR * death_rate
                    projected_hrs = [hr + datetime.timedelta(days=death_lag) for hr in mdl_hours]
                elif prediction_mode == 'exponential':  # draw delays from exponential distribution
                    _, mdl_hourly_new_deaths = draw_cases_and_deaths_from_exponential_distribution(
                        mdl_IR, detection_rate, detection_lag, death_rate, death_lag)
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
                elif prediction_mode == 'gamma':  # draw delays from gamma distribution
                    _, mdl_hourly_new_deaths = draw_cases_and_deaths_from_gamma_distribution(
                        mdl_IR, detection_rate, death_rate)
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours
                else:  # modeled confirmed deaths during simulation
                    assert 'new_confirmed_cases' in history[subset]
                    mdl_hourly_new_deaths = history[subset]['new_deaths']
                    mdl_prediction = get_cumulative(mdl_hourly_new_deaths)
                    projected_hrs = mdl_hours

            if not make_plot: #default
                # note: y_pred is also cumulative, but represents seed x day, instead of hour
                y_true, y_pred, eval_start, eval_end = find_model_and_real_overlap_for_eval(
                    real_dates, real_data, projected_hrs, mdl_prediction, 
                    compare_start_time, compare_end_time)
                if len(y_true) < 5:
                    print("Fewer than 5 days of overlap between model predictions and observed %s data; not scoring" % mode)
                else:
                    score_dict['eval_start_time_%s' % mode] = eval_start
                    score_dict['eval_end_time_%s' % mode] = eval_end
                    score_dict['cumulative_predicted_%s' % mode] = y_pred
                    score_dict['cumulative_true_%s' % mode] = y_true
                    score_dict['cumulative_%s_RMSE' % mode] = compute_loss(
                        y_true, y_pred, metric='RMSE', min_threshold=None, 
                        compare_daily_not_cumulative=False)
                    score_dict['cumulative_%s_MSE' % mode] = compute_loss(
                        y_true, y_pred, metric='MSE', min_threshold=None, 
                        compare_daily_not_cumulative=False)

                    # the following checks are to deal with converting a cumulative curve back to a daily
                    # curve when the eval starts past the first day, which means the first entry in the
                    # cumulative curve is already an accumulation from multiple days, so we need to subtract
                    # the cumulative value from the previous day
                    if eval_start > real_dates[0]:
                        eval_start_index = real_dates.index(eval_start)
                        cumulative_day_before = real_data[eval_start_index-1]
                        y_true = y_true - cumulative_day_before
                    if eval_start >= projected_hrs[24]:  # starting eval on day 2+ of simulation
                        eval_start_index = projected_hrs.index(eval_start)
                        cumulative_day_before = mdl_prediction[:, eval_start_index-24]
                        y_pred = (y_pred.T - cumulative_day_before).T
                    # note: all metrics below this should be computed on *daily* cases / deaths per day, not cumulative
                    score_dict['daily_%s_RMSE' % mode] = compute_loss(
                        y_true, y_pred, metric='RMSE', min_threshold=None, compare_daily_not_cumulative=True)
                    score_dict['daily_%s_MSE' % mode] = compute_loss(
                        y_true, y_pred, metric='MSE', min_threshold=None, compare_daily_not_cumulative=True)

                    if prediction_mode == 'deterministic':  # LL metrics assume constant delay and rate for predictions
                        threshold_metrics = [
                            'MRE',
                            'poisson_NLL']
                        rate = detection_rate if mode == 'cases' else death_rate
                        for threshold_metric in threshold_metrics:
                            for min_threshold in min_thresholds:
                                for do_logsumexp in [True, False]:
                                    if do_logsumexp:
                                        agg_str = 'logsumexp'
                                    else:
                                        agg_str = 'sum'

                                    # Skip logsumexp for MRE since it has no LL interpretation
                                    if threshold_metric == 'MRE' and do_logsumexp:
                                        continue

                                    dict_str = f'daily_{mode}_{threshold_metric}_thres-{min_threshold}_{agg_str}'
                                    score_dict[dict_str] = compute_loss(
                                        y_true=y_true,
                                        y_pred=y_pred,
                                        rate=rate,
                                        metric=threshold_metric,
                                        min_threshold=min_threshold,
                                        compare_daily_not_cumulative=True,
                                        do_logsumexp=do_logsumexp)

        if return_mdl_pred_and_hours and plot_mode == mode:
            return (mdl_prediction, projected_hrs),[]

        if make_plot and plot_mode == mode:
            assert(ax is not None and title is not None)
            if plot_daily_not_cumulative:
                new_projected_hrs = []
                new_mdl_prediction = []
                for hr, prediction in zip(projected_hrs, mdl_prediction.T):
                    if hr.hour == 0:
                        new_projected_hrs.append(hr)
                        new_mdl_prediction.append(prediction)
                projected_hrs = new_projected_hrs
                mdl_prediction = np.array(new_mdl_prediction).T
                mdl_prediction = get_daily_from_cumulative(mdl_prediction)
                real_data = get_daily_from_cumulative(real_data)

            num_seeds, _ = mdl_prediction.shape
            if num_seeds > 1:
                mean, lower_CI, upper_CI = mean_and_CIs_of_timeseries_matrix(mdl_prediction, alpha=.05)
                #mean, lower_CI, upper_CI = mean_and_CIs_of_timeseries_matrix(mdl_prediction)
                model_max = max(upper_CI)
            else:
                mean = mdl_prediction[0]
                model_max = max(mean)
            real_max = max(real_data)
            daily_or_cumulative_string = 'daily' if plot_daily_not_cumulative else 'cumulative'
            if model_line_label is None:
                model_line_label = 'modeled %s %s' % (daily_or_cumulative_string, mode)
            if true_line_label is None:
                true_line_label = 'true %s %s' % (daily_or_cumulative_string, mode)
            ax.plot_date(projected_hrs, mean, linestyle='-', label=model_line_label, c=model_color,
                         markersize=marker_size)
            if plot_real_data:
                if plot_daily_not_cumulative:
                    # use non-connected x's if plotting non-smoothed daily cases / deaths
                    if add_smoothed_real_data_line:
                        smoothed_real_data = apply_smoothing(real_data, before=3, after=3)
                        ax.plot_date(
                            real_dates, smoothed_real_data, linestyle='-',
                            label=true_line_label, c=real_data_color, markersize=marker_size)
                        ax.plot_date(
                            real_dates, real_data, marker='x', c='grey', alpha=0.8,
                            markersize=marker_size+1, markeredgewidth=2)
                    else:
                        smoothed_real_data = None
                        ax.plot_date(
                            real_dates, real_data, label=true_line_label, marker='x', 
                            c=real_data_color, markersize=marker_size+1, markeredgewidth=2)
            if num_seeds > 1 and plot_errorbars:
                ax.fill_between(
                    projected_hrs, lower_CI, upper_CI, alpha=.2, color=model_color)

            interval = int(len(real_dates) / 6)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if only_plot_intersection:
                ax.set_xlim([max(min(projected_hrs), min(real_dates)), min(max(projected_hrs), max(real_dates))]) # only plot place where both lines intersect.
                right = min(max(projected_hrs), max(real_dates))
                model_max_idx = projected_hrs.index(right)
                if num_seeds > 1:
                    model_max = max(upper_CI[:model_max_idx])
                else:
                    model_max = max(mean[:model_max_idx])
                real_max_idx = real_dates.index(right)
                real_max = max(real_data[:real_max_idx])

            if plot_log:
                ax.set_yscale('log')
                ax.set_ylim([1, max(model_max, real_max)])
            else:
                ax.set_ylim([0, max(model_max, real_max)])

            if plot_legend:
                ax.legend(fontsize=legend_fontsize, loc='upper left')

            if xticks is None:
                if x_interval is None:
                    x_interval = int(len(real_dates) / 6)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_interval))
            else:
                ax.set_xticks(xticks)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.tick_params(labelsize=tick_label_fontsize)
            if y_range is not None:
                ax.set_ylim(*y_range)
            if x_range is not None:
                ax.set_xlim(*x_range)

            if only_two_yticks:

                bot, top = ax.get_ylim()
                if plot_mode == 'cases':
                    # Round to nearest hundred
                    top = (top // 100) * 100
                elif plot_mode == 'deaths':
                    # Round to nearest 20
                    top = (top // 20) * 20
                ax.set_yticks([bot, top])

            if plot_mode == 'cases':
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else '{:.1f}'.format(x/1000) + 'k'))

            ax.grid(alpha=.5)
            ax.set_title(title, fontsize=title_fontsize)

    #new_data_to_return = [real_dates, mean, real_data, smoothed_real_data, projected_hrs, lower_CI, upper_CI]
    new_data_to_return = [real_dates, real_data, projected_hrs, mdl_prediction]    
    return score_dict, new_data_to_return


def evaluate_all_fitted_models_for_experiment_subset(
    experiment_to_run,
    subset,
    min_timestring=None,
    max_timestring=None,
    timestrings=None,
    required_properties=None,
    required_model_kwargs=None,
    required_data_kwargs=None,
    result_types=None,
    key_to_sort_by=None,
    old_directory=False):
    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs`
    """
    if required_properties is None:
        required_properties = {}
    required_properties['experiment_to_run'] = experiment_to_run
    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_data_kwargs is None:
        required_data_kwargs = {}

    if timestrings is None:
        timestrings = filter_timestrings_for_properties(
            required_properties=required_properties,
            required_model_kwargs=required_model_kwargs,
            required_data_kwargs=required_data_kwargs,
            min_timestring=min_timestring,
            max_timestring=max_timestring,
            old_directory=old_directory)
        print('Found %d fitted models for %s' % (len(timestrings), experiment_to_run))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    if result_types is None:
        result_types = ['loss_dict', 'train_loss_dict', 'test_loss_dict','estimated_R0']
        # ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']
    results = []
    start_time = time.time()
    for i, ts in enumerate(timestrings):
        
        _, kwargs, _, model_results, fast_to_load_results = get_subset_model_and_data_from_timestring(
            ts, subset,
            verbose=False, 
            load_fast_results_only=True, 
            old_directory=old_directory)
        
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']

        results_for_ts = {
            'timestring':ts,
            'data_kwargs':data_kwargs,
            'model_kwargs':model_kwargs,
            'results':model_results,
            'experiment_to_run':experiment_to_run
        }

        if 'final infected fraction' in fast_to_load_results:
            results_for_ts['final infected fraction'] = fast_to_load_results['final infected fraction']

        for result_type in result_types:
            if (result_type in fast_to_load_results) and (fast_to_load_results[result_type] is not None):
                for k in fast_to_load_results[result_type]:
                    full_key = result_type + '_' + k
                    assert full_key not in results_for_ts
                    results_for_ts[full_key] = fast_to_load_results[result_type][k]

        for k in exo_kwargs:
            assert k not in results_for_ts
            results_for_ts[k] = exo_kwargs[k]
        for k in model_kwargs:
            if k == 'exogenous_model_kwargs':
                continue
            else:
                assert k not in results_for_ts
                results_for_ts[k] = model_kwargs[k]
        results.append(results_for_ts)
        if i % 1000 == 0:
            curr_time = time.time()
            print('Loaded %d models so far: %.3fs -> %.3fs per model' %
                  (len(results), curr_time-start_time, (curr_time-start_time)/len(results)))

    end_time = time.time()
    print('Time to load and score all models: %.3fs -> %.3fs per model' %
          (end_time-start_time, (end_time-start_time)/len(timestrings)))
    results = pd.DataFrame(results)
    
    #NEW
    results['subset'] = subset
    
    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results


def get_fast_to_load_results_subset(subset, kwargs, fitted_model):
    model_kwargs = kwargs['model_kwargs']
    # exo_kwargs = model_kwargs['exogenous_model_kwargs']
    data_kwargs = kwargs['data_kwargs']
    experiment_to_run = kwargs['experiment_to_run']

    if 'grid_search' in experiment_to_run:
        train_test_partition = TRAIN_TEST_PARTITION
    else:
        train_test_partition = None

    # TODO: understand what is needed from here and load it in other ways
    # nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(
    #     data_kwargs['MSA_name'])
    if subset == 'all' or subset == 'nyt':
        nyt_outcomes = get_variables_for_evaluating_province_model(data_kwargs['MSA_name'])
    else:
        nyt_outcomes = get_variables_for_evaluating_province_model(subset)

    #----- 4b. TRY to save separate model results (see what fails here) --------
    model_results_to_save_separately = {}
    #for attr_to_save_separately in ['history', 'CBGS_TO_IDXS']:
    for attr_to_save_separately in ['history']:
        model_results_to_save_separately[attr_to_save_separately] = getattr(
            fitted_model, attr_to_save_separately)

    #----- 4c. ALWAYS Save separate descriptive results --------
    # evaluate model fit to cases and save loss separately as well.
    # Everything saved in this data structure should be a summary result - small and fast to load, numbers only!
    #print("# ---- Estimating losses - 1. all sample")
    loss_dict, _ = compare_model_vs_real_num_cases_subset(
        nyt_outcomes,
        model_kwargs['min_datetime'],
        subset=subset,
        model_results=model_results_to_save_separately)

    # fast_to_load_results initialized  here
    fast_to_load_results = {'loss_dict': loss_dict}

    if train_test_partition is not None:
        #print("# ---- Estimating losses - 2 train set")
        train_max = train_test_partition + datetime.timedelta(hours=-1)
        #print(train_test_partition)
        #print(train_max)
        train_loss_dict, _ = compare_model_vs_real_num_cases_subset(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            subset=subset,
            compare_end_time = train_max,
            model_results=model_results_to_save_separately)       
        fast_to_load_results['train_loss_dict'] = train_loss_dict
        #print(train_loss_dict)

        #print("# ---- Estimating losses - 3 test set")
        test_loss_dict, _ = compare_model_vs_real_num_cases_subset(
            nyt_outcomes,
            model_kwargs['min_datetime'],
            subset=subset,
            compare_start_time = train_test_partition,
            model_results=model_results_to_save_separately)
        fast_to_load_results['test_loss_dict'] = test_loss_dict
        #print(train_loss_dict)

        fast_to_load_results['train_test_date_cutoff'] = train_test_partition
        sanity_check_error_metrics(fast_to_load_results)

    fast_to_load_results['clipping_monitor'] = fitted_model.clipping_monitor
    final_infected_fraction = (
        fitted_model.cbg_infected + fitted_model.cbg_removed + fitted_model.cbg_latent).sum(axis=1)/fitted_model.CBG_SIZES.sum()
    fast_to_load_results['final infected fraction'] = final_infected_fraction    
    fast_to_load_results['estimated_R0'] = fitted_model.estimated_R0
    fast_to_load_results['intervention_cost'] = fitted_model.INTERVENTION_COST
    return fast_to_load_results

def get_subset_model_and_data_from_timestring(
    timestring, subset,
    verbose=False, 
    load_original_data=False,
    #load_full_model=False, 
    load_fast_results_only=True,
    #load_filtered_data_model_was_fitted_on=False,
    old_directory=False):
    if verbose:
        print("Loading model from timestring %s" % timestring)
    if old_directory:
        model_dir = OLD_FITTED_MODEL_DIR
    else:
        model_dir = FITTED_MODEL_DIR
    
    with open(os.path.join(model_dir, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb') as f:
        data_and_model_kwargs = pickle.load(f)
    
    model_results = None
    # if SAVE_MODEL_RESULTS_SEPARATELY:
    #     f = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'rb')
    #     model_results = pickle.load(f)
    #     f.close()    
    
    filepath = os.path.join(model_dir, 'full_models', 'fitted_model_%s.gzip' % timestring)
    with gzip.open(filepath, "rb") as f:
        model = pickle.load(f)        
    if verbose:    
        print("Loading model at %s..." % filepath)

    d = None
    if load_original_data:
        if verbose:
            print("Loading original data as well...warning, this may take a while")
        d = helper.load_dataframe_for_individual_msa(**data_and_model_kwargs['data_kwargs'])    

    # Change this
    fast_to_load_results = get_fast_to_load_results_subset(subset, data_and_model_kwargs, model)   

    return model, data_and_model_kwargs, d, model_results, fast_to_load_results    

def plot_best_models_fit_for_msa_subset(
    df, msa_name, ax, key_to_sort_by, train_test_partition,
    plotting_kwargs, 
    subset,
    old_directory=False, 
    threshold=ACCEPTABLE_LOSS_TOLERANCE):

    subdf = df[(df['MSA_name'] == msa_name)].copy()
    subdf = subdf.sort_values(by=key_to_sort_by)
    losses = subdf[key_to_sort_by] / subdf.iloc[0][key_to_sort_by]
    num_models_to_aggregate = np.sum(losses <= threshold)
    try:
        assert num_models_to_aggregate <= MAX_MODELS_TO_TAKE_PER_MSA
    except AssertionError:
        num_models_to_aggregate = MAX_MODELS_TO_TAKE_PER_MSA
        print(f"Warning: too many best fit models. Restricting their number to {MAX_MODELS_TO_TAKE_PER_MSA}.")
        
    #print('Found %d best fit models within threshold for %s' % (num_models_to_aggregate, MSAS_TO_PRETTY_NAMES[msa_name]))
    # NEW_V2
    print('Found %d best fit models within threshold for %s' % (num_models_to_aggregate, msa_name))
    
    # Aggregate predictions from best fit models that are within the ACCEPTABLE_LOSS_TOLERANCE
    mdl_predictions = []
    old_projected_hrs = None
    individual_plotting_kwargs = plotting_kwargs.copy()
    individual_plotting_kwargs['return_mdl_pred_and_hours'] = True  # don't plot individual models
    for model_idx in range(num_models_to_aggregate):
        ts = subdf.iloc[model_idx]['timestring']
        model, kwargs, _, _, _ = get_subset_model_and_data_from_timestring(
            ts, subset, old_directory=old_directory)
        model_kwargs = kwargs['model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        (mdl_prediction, projected_hrs),_ = plot_model_fit_from_model_and_kwargs_subset(
            ax,
            model_kwargs,
            data_kwargs,
            subset=subset,
            model=model,
            plotting_kwargs=individual_plotting_kwargs,
            train_test_partition=train_test_partition)
        mdl_predictions.append(mdl_prediction)
        if old_projected_hrs is not None:
            assert projected_hrs == old_projected_hrs
        old_projected_hrs = projected_hrs
    mdl_predictions = np.concatenate(mdl_predictions, axis=0)

    # Plot aggregate predictions
    agg_plotting_kwargs = plotting_kwargs.copy()
    agg_plotting_kwargs['mdl_prediction'] = mdl_predictions
    agg_plotting_kwargs['projected_hrs'] = projected_hrs
    _, new_data_to_return = plot_model_fit_from_model_and_kwargs_subset(
        ax,
        model_kwargs,
        data_kwargs,
        subset=subset,
        plotting_kwargs=agg_plotting_kwargs,
        train_test_partition=train_test_partition,
    )
    ax.grid(False)
    return new_data_to_return

def plot_model_fit_from_model_and_kwargs_subset(ax,
                                         mdl_kwargs,
                                         data_kwargs,
                                         subset,
                                         model=None,
                                         train_test_partition=None,
                                         model_results=None,
                                         plotting_kwargs=None):
    msa_name = data_kwargs['MSA_name']
    #nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)    
    
    if subset == 'all' or subset == 'nyt':
        nyt_outcomes = get_variables_for_evaluating_province_model(msa_name)
    else:
        nyt_outcomes = get_variables_for_evaluating_province_model(subset)

    min_datetime = mdl_kwargs['min_datetime']
    if plotting_kwargs is None:
        plotting_kwargs = {}  # could include options like plot_mode, plot_log, etc.
    if 'title' not in plotting_kwargs:
        plotting_kwargs['title'] = MSAS_TO_PRETTY_NAMES[msa_name]
    if 'make_plot' not in plotting_kwargs:
        plotting_kwargs['make_plot'] = True
    ret_val, new_data_to_return = compare_model_vs_real_num_cases_subset(
        nyt_outcomes, 
        min_datetime,
        subset=subset,
        model=model,
        model_results=model_results,
        ax=ax,
        **plotting_kwargs)
    if train_test_partition is not None and plotting_kwargs['make_plot']:
        ax.plot_date([train_test_partition, train_test_partition], ax.get_ylim(), color='black', linestyle='-')
    return ret_val, new_data_to_return        