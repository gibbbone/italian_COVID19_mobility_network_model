from covid_constants_and_util import *
from utilities import *
import matplotlib.ticker as ticker
import helper_methods_for_aggregate_data_analysis as helper
from collections import Counter
from scipy.special import logsumexp
import pickle
import time 
import math 
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gzip
#from run_one_model import load_model_and_data_from_timestring
from utilities import *

def sanity_check_error_metrics(fast_to_load_results):
    """
    Make sure train and test loss sum to total loss in the way we would expect.
    """
    n_train_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['train_loss_dict']['eval_start_time_cases'],
        fast_to_load_results['train_loss_dict']['eval_end_time_cases']))

    n_test_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['test_loss_dict']['eval_start_time_cases'],
        fast_to_load_results['test_loss_dict']['eval_end_time_cases']))

    n_total_days = len(helper.list_datetimes_in_range(
        fast_to_load_results['loss_dict']['eval_start_time_cases'],
        fast_to_load_results['loss_dict']['eval_end_time_cases']))

    assert n_train_days + n_test_days == n_total_days
    assert fast_to_load_results['loss_dict']['eval_end_time_cases'] == fast_to_load_results['test_loss_dict']['eval_end_time_cases']
    assert fast_to_load_results['loss_dict']['eval_start_time_cases'] == fast_to_load_results['train_loss_dict']['eval_start_time_cases']
    for key in ['daily_cases_MSE', 'cumulative_cases_MSE']:
        if 'RMSE' in key:
            train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] ** 2 +
                 n_test_days * fast_to_load_results['test_loss_dict'][key] ** 2)

            overall_loss = n_total_days * fast_to_load_results['loss_dict'][key] ** 2
        else:
            train_plus_test_loss = (n_train_days * fast_to_load_results['train_loss_dict'][key] +
                 n_test_days * fast_to_load_results['test_loss_dict'][key])

            overall_loss = n_total_days * fast_to_load_results['loss_dict'][key]

        assert np.allclose(train_plus_test_loss, overall_loss, rtol=1e-6)
    print("Sanity check error metrics passed")

#########################################################
# Functions to evaluate model fit and basic results
#########################################################
def plot_slir_over_time(mdl,
    ax,
    plot_logarithmic=True,
    timesteps_to_plot=None,
    groups_to_plot=None,
    lines_to_plot=None,
    title=None):
    """
    Plot SLIR fractions over time.
    """
    if groups_to_plot is None:
        groups_to_plot = ['all']
    history = copy.deepcopy(mdl.history)
    for group in history.keys():
        history[group]['L+I+R'] = history[group]['latent'] + history[group]['infected'] + history[group]['removed']

    if lines_to_plot is None:
        lines_to_plot = ['susceptible', 'latent', 'infected', 'removed']

    linestyles = ['-', '--', '-.', ':']
    colors = ['black', 'orange', 'blue', 'green', 'red']
    lines_to_return = {}

    for line_idx, k in enumerate(lines_to_plot):
        for group_idx, group in enumerate(groups_to_plot):
            total_population = history[group]['total_pop']
            time_in_days = np.arange(history[group][k].shape[1]) / 24.
            x = time_in_days
            y = (history[group][k].T / total_population).T
            assert y.shape[1] == x.shape[0]
            mean_Y, lower_CI_Y, upper_CI_Y = mean_and_CIs_of_timeseries_matrix(y)
            assert len(mean_Y) == len(x)

            color = colors[line_idx % len(colors)]
            linestyle = linestyles[group_idx % len(linestyles)]
            n_cbgs = history[group]['num_cbgs']
            if timesteps_to_plot is not None:
                x = x[:timesteps_to_plot]
                mean_Y = mean_Y[:timesteps_to_plot]
                lower_CI_Y = lower_CI_Y[:timesteps_to_plot]
                upper_CI_Y = upper_CI_Y[:timesteps_to_plot]

            states_to_legend_labels = {
                'latent':'E (exposed)',
                'infected':'I (infectious)',
                'removed':'R (removed)',
                'susceptible':'S (susceptible)',
                'L+I+R':'E+I+R'
            }
            if group != 'all':
                ax.plot(x, mean_Y, label='%s, %s' % (states_to_legend_labels[k], group), color=color, linestyle=linestyle)
            else:
                ax.plot(x, mean_Y, label='%s' % (states_to_legend_labels[k]), color=color, linestyle=linestyle)
            ax.fill_between(x, lower_CI_Y, upper_CI_Y, color=color, alpha=.2)

            if plot_logarithmic:
                ax.set_yscale('log')

            lines_to_return['%s, %s' % (k, group)] = mean_Y
    ax.legend(fontsize=16) # Removed for now because we need to handle multiple labels
    logarithmic_string = ' (logarithmic)' if plot_logarithmic else ''
    ax.set_xlabel('Time (in days)', fontsize=16)
    ax.set_ylabel("Fraction of population%s" % logarithmic_string, fontsize=16)
    ax.set_xticks(range(0, math.ceil(max(time_in_days)) + 1, 7))
    plt.xlim(0, math.ceil(max(time_in_days)))
    if plot_logarithmic:
        ax.set_ylim([1e-6, 1])
    else:
        ax.set_ylim([-.01, 1])
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=.5)
    return lines_to_return

def make_slir_plot_stratified_by_demographic_attribute(
    mdl, ax, attribute, median_or_decile,slir_lines_to_plot=None):
    """
    Given a demographic attribute, plot SLIR curves for people above and below median
    if median_or_decile = median, or top and bottom decile, if median_or_decile = decile.
    """
    if slir_lines_to_plot is None:
        slir_lines_to_plot = ['L+I+R']
    assert attribute in ['p_black', 'p_white', 'median_household_income']

    if median_or_decile not in ['median', 'decile', 'above_median_in_own_county']:
        raise Exception("median_or_decile should be 'median' or 'decile' or 'above_median_in_own_county'")
    if median_or_decile == 'median':
        groups_to_plot = [f'{attribute}_above_median', f'{attribute}_below_median']
        title = 'above and below median for %s' % attribute
    elif median_or_decile == 'decile':
        groups_to_plot = [f'{attribute}_top_decile', f'{attribute}_bottom_decile']
        title = 'top and bottom decile for %s' % attribute
    elif median_or_decile == 'above_median_in_own_county':
        groups_to_plot = [f'{attribute}_above_median_in_own_county', f'{attribute}_below_median_in_own_county']
        title = 'above and below COUNTY median for %s' % attribute

    if attribute != 'p_black':
        groups_to_plot = groups_to_plot[::-1] # keep underserved population consistent. Should always be solid line (first line plotted)

    lines_to_return = plot_slir_over_time(
        mdl,
        ax,
        groups_to_plot=groups_to_plot,
        lines_to_plot=slir_lines_to_plot,
        title=title)
    return lines_to_return

def make_slir_race_ses_plot(mdl, path_to_save=None, title_string=None, slir_lines_to_plot=None):
    """
    Plot SLIR curves stratified by race and SES.
    Returns a dictionary which stores the values for each SLIR curve.
    """
    all_results = {}
    fig = plt.figure(figsize=[30, 20])
    subplot_idx = 1
    for demographic_attribute in ['p_black', 'p_white', 'median_household_income']:
        for median_or_decile in ['median', 'decile', 'above_median_in_own_county']:
            ax = fig.add_subplot(3, 3, subplot_idx)
            results = make_slir_plot_stratified_by_demographic_attribute(
                mdl=mdl,
                ax=ax,
                attribute=demographic_attribute,
                median_or_decile=median_or_decile,
                slir_lines_to_plot=slir_lines_to_plot)
            for k in results:
                assert k not in all_results
                all_results[k] = results[k]
            subplot_idx += 1
    if title_string is not None:
        fig.suptitle(title_string)
    if path_to_save is not None:
        fig.savefig(path_to_save)
    else:
        plt.show()
    return all_results

def get_datetimes_and_totals_from_nyt_outcomes(nyt_outcomes):
    """
    Adapted to Italian data: no death count.
    """
    date_groups = nyt_outcomes.groupby('date').indices
    dates = sorted(date_groups.keys())
    datetimes = []
    total_cases = []
    #total_deaths = []
    for date in dates:
        year, month, day = date.split('-')
        curr_datetime = datetime.datetime(int(year), int(month), int(day))
        if len(datetimes) > 0:
            assert(curr_datetime > datetimes[-1])
        datetimes.append(curr_datetime)
        rows = nyt_outcomes.iloc[date_groups[date]]
        total_cases.append(np.sum(rows['cases'].to_numpy()))
        #total_deaths.append(np.sum(rows['deaths'].to_numpy()))
    return datetimes, np.array(total_cases)#, np.array(total_deaths)


def get_datetimes_and_totals_from_nyt_outcomes_OLD(nyt_outcomes):
    # nyt_outcomes contains different counties data
    # here we are merging all counties from the same MSA
    date_groups = nyt_outcomes.groupby('date').indices
    dates = sorted(date_groups.keys())
    datetimes = []
    total_cases = []
    total_deaths = []
    for date in dates:
        year, month, day = date.split('-')
        curr_datetime = datetime.datetime(int(year), int(month), int(day))
        if len(datetimes) > 0:
            assert(curr_datetime > datetimes[-1])
        datetimes.append(curr_datetime)
        rows = nyt_outcomes.iloc[date_groups[date]]
        total_cases.append(np.sum(rows['cases'].to_numpy()))
        total_deaths.append(np.sum(rows['deaths'].to_numpy()))
    return datetimes, np.array(total_cases), np.array(total_deaths)

def find_model_and_real_overlap_for_eval(
    real_dates, real_cases, mdl_hours, mdl_cases,
    compare_start_time=None, compare_end_time=None):

    # NEW
    real_dates = [rd for rd in real_dates if rd <= MAX_DATETIME]

    overlap = set(real_dates).intersection(set(mdl_hours))
    if compare_start_time is None:
        compare_start_time = min(overlap)
    if compare_end_time is None:
        compare_end_time = max(overlap)

    # NEW_V2
    # assert type(mdl_start_time) == datetime.datetime
    # assert type(mdl_end_time) == datetime.datetime
    # print(mdl_start_time, mdl_end_time, mdl_start_time<mdl_end_time)    
    # print(compare_start_time, compare_end_time, compare_start_time < compare_end_time)

    comparable_period = helper.list_hours_in_range(compare_start_time, compare_end_time)
    overlap = sorted(overlap.intersection(set(comparable_period)))
    real_date2case = dict(zip(real_dates, real_cases))
    mdl_date2case = dict(zip(mdl_hours, mdl_cases.T)) #mdl_cases has an extra random_seed first dim
    real_vec = []
    mdl_mat = np.zeros((len(mdl_cases), len(overlap)))  # num_seed x num_time
    for idx, date in enumerate(overlap):
        real_vec.append(real_date2case[date])
        mdl_mat[:, idx] = mdl_date2case[date]
    return np.array(real_vec), mdl_mat, overlap[0], overlap[-1]

def resave_fast_to_load_results_for_timestring(ts, old_directory, nyt_outcomes):
    """
    Overwrite old loss if we want to add additional features.
    """
    t0 = time.time()
    model, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(
         ts,
         verbose=False,
         load_fast_results_only=False,
         load_full_model=True,
         old_directory=old_directory)
    model_kwargs = kwargs['model_kwargs']
    data_kwargs = kwargs['data_kwargs']
    train_test_partition = fast_to_load_results['train_test_date_cutoff']
    keys_to_rewrite = ['loss_dict', 'train_loss_dict', 'test_loss_dict']


    for key_to_rewrite in keys_to_rewrite:
        old_loss_dict = None
        new_loss_dict = None
        old_loss_dict = fast_to_load_results[key_to_rewrite]
        if key_to_rewrite == 'loss_dict':
            new_loss_dict,_ = compare_model_vs_real_num_cases(nyt_outcomes,
                                               model_kwargs['min_datetime'],
                                               model=model,
                                               make_plot=False)
        elif key_to_rewrite == 'train_loss_dict':
            train_max = train_test_partition + datetime.timedelta(hours=-1)
            new_loss_dict,_ = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_end_time = train_max,
                                           model=model)
        elif key_to_rewrite == 'test_loss_dict':
            new_loss_dict,_ = compare_model_vs_real_num_cases(nyt_outcomes,
                                           model_kwargs['min_datetime'],
                                           compare_start_time = train_test_partition,
                                           model=model)

        common_keys = [a for a in new_loss_dict.keys() if a in old_loss_dict.keys()]
        assert len(common_keys) > 0
        for k in common_keys:
            if type(new_loss_dict[k]) is not np.ndarray:
                assert new_loss_dict[k] == old_loss_dict[k]
            else:
                assert np.allclose(new_loss_dict[k], old_loss_dict[k])

        fast_to_load_results[key_to_rewrite] = new_loss_dict

    if old_directory:
        model_dir = OLD_FITTED_MODEL_DIR
    else:
        model_dir = FITTED_MODEL_DIR
    path_to_save = os.path.join(model_dir, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % ts)
    assert os.path.exists(path_to_save)
    file = open(path_to_save, 'wb')
    pickle.dump(fast_to_load_results, file)
    file.close()
    print("Time to save model: %2.3f seconds" % (time.time() - t0))

def compare_model_vs_real_num_cases(
    nyt_outcomes,
    mdl_start_time,
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
    model_color='darkorchid',#'darkorchid',
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
        assert('nyt' in history)
        assert model_results is None
        assert mdl_prediction is None
        assert projected_hrs is None
    elif model_results is not None: #default
        # NEW_V2
        #cbgs_to_idxs = model_results['CBGS_TO_IDXS']
        history = model_results['history']
        assert('nyt' in history)
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
        mdl_IR = (history['nyt']['infected'] + history['nyt']['removed']) 
        num_hours = mdl_IR.shape[1]
        mdl_end_time = mdl_start_time + datetime.timedelta(hours=num_hours-1)
               
        mdl_hours = helper.list_hours_in_range(mdl_start_time, mdl_end_time)
        mdl_dates = helper.list_datetimes_in_range(mdl_start_time, mdl_end_time)
        assert(mdl_start_time < mdl_end_time)
    else:
        mdl_IR = None

    #modes = ['cases', 'deaths']
    modes = ['cases']
    print("Warning: data validation done only on cases, because we do not have data on deaths at the province level in Italian data.")
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
                    assert 'new_confirmed_cases' in history['nyt']
                    mdl_hourly_new_cases = history['nyt']['new_confirmed_cases']
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
                    assert 'new_confirmed_cases' in history['nyt']
                    mdl_hourly_new_deaths = history['nyt']['new_deaths']
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
            return mdl_prediction, projected_hrs

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
                        ax.plot_date(real_dates, smoothed_real_data, linestyle='-',
                                     label=true_line_label, c=real_data_color, markersize=marker_size)
                        ax.plot_date(real_dates, real_data, marker='x', c='grey', alpha=0.8,
                                     markersize=marker_size+1, markeredgewidth=2)
                    else:
                        ax.plot_date(real_dates, real_data, label=true_line_label, marker='x', c=real_data_color, markersize=marker_size+1, markeredgewidth=2)
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
    new_data_to_return = [real_dates, real_data, projected_hrs, mdl_prediction]    
    return score_dict, new_data_to_return

def draw_cases_and_deaths_from_exponential_distribution(model_IR, detection_rate, detection_lag_in_days,
                                                        death_rate, death_lag_in_days, random_seed=0):
    # model_IR should be a matrix of seed x hour, where each entry represents the *cumulative* number
    # of people in infectious or removed for that seed and hour
    # eg mdl_IR = (model.history['nyt']['infected'] + model.history['nyt']['removed'])
    np.random.seed(random_seed)
    detection_lag = detection_lag_in_days * 24  # want the lags in hours
    death_lag = death_lag_in_days * 24
    num_seeds, num_hours = model_IR.shape
    assert num_hours % 24 == 0
    hourly_new_infectious = get_daily_from_cumulative(model_IR)

    predicted_cases = np.zeros((num_seeds, num_hours))
    predicted_deaths = np.zeros((num_seeds, num_hours))
    cases_to_confirm = np.zeros(num_seeds)
    deaths_to_happen = np.zeros(num_seeds)
    for hr in range(num_hours):
        new_infectious = hourly_new_infectious[:, hr]
        new_confirmed_cases = np.random.binomial(cases_to_confirm.astype(int), 1/detection_lag)
        predicted_cases[:, hr] = new_confirmed_cases
        new_cases_to_confirm = np.random.binomial(new_infectious.astype(int), detection_rate)
        cases_to_confirm = cases_to_confirm + new_cases_to_confirm - new_confirmed_cases
        new_deaths = np.random.binomial(deaths_to_happen.astype(int), 1/death_lag)
        predicted_deaths[:, hr] = new_deaths
        new_deaths_to_happen = np.random.binomial(new_infectious.astype(int), death_rate)
        deaths_to_happen = deaths_to_happen + new_deaths_to_happen - new_deaths
    return predicted_cases, predicted_deaths

def draw_cases_and_deaths_from_gamma_distribution(
    model_IR, detection_rate, death_rate,
    detection_delay_shape=1.85,  # Li et al. (Science 2020)
    detection_delay_scale=3.57,
    death_delay_shape=1.85,
    death_delay_scale=9.72,
    random_seed=0):
    """
    Model_IR should be a matrix of seed x hour, where each entry represents the *cumulative* number
    of people in infectious or removed for that seed and hour
    eg mdl_IR = (model.history['nyt']['infected'] + model.history['nyt']['removed'])    
    """    
    np.random.seed(random_seed)
    num_seeds, num_hours = model_IR.shape
    assert num_hours % 24 == 0
    hourly_new_infectious = get_daily_from_cumulative(model_IR)

    predicted_cases = np.zeros((num_seeds, num_hours))
    predicted_deaths = np.zeros((num_seeds, num_hours))
    for hr in range(num_hours):
        new_infectious = hourly_new_infectious[:, hr]  # 1 x S
        cases_to_confirm = np.random.binomial(new_infectious.astype(int), detection_rate)
        deaths_to_happen = np.random.binomial(new_infectious.astype(int), death_rate)
        for seed in range(num_seeds):
            num_cases = cases_to_confirm[seed]
            confirmation_delays = np.random.gamma(detection_delay_shape, detection_delay_scale, int(num_cases))
            confirmation_delays = confirmation_delays * 24  # convert delays from days to hours
            counts = Counter(confirmation_delays).most_common()
            for delay, count in counts:
                projected_hr = int(hr + delay)
                if projected_hr < num_hours:
                    predicted_cases[seed, projected_hr] = predicted_cases[seed, projected_hr] + count

            num_deaths = deaths_to_happen[seed]
            death_delays = np.random.gamma(death_delay_shape, death_delay_scale, int(num_deaths))
            death_delays = death_delays * 24  # convert delays from days to hours
            counts = Counter(death_delays).most_common()
            for delay, count in counts:
                projected_hr = int(hr + delay)
                if projected_hr < num_hours:
                    predicted_deaths[seed, projected_hr] = predicted_deaths[seed, projected_hr] + count
    return predicted_cases, predicted_deaths

def compute_loss(y_true, y_pred, rate=None,
                 metric='RMSE',
                 min_threshold=None,
                 compare_daily_not_cumulative=True,
                 do_logsumexp=False):
    """
    (This assumes that y_true and y_pred are cumulative counts).
    
    --- Input ---
    y_true: 1D array
        The true case/death counts
    y_pred: 2D array
        The predicted case/death counts over all seeds
    rate: 
        The detection or death rate used in computing y_pred; only required when metric is 
        poisson_NLL
    metric:  str
        RMSE or MRE, the loss metric
    min_threshold: 
        The minimum number of true case/deaths that a day must have to be included in eval
    compare_daily_not_cumulative: bool
        Converts y_true and y_pred into daily counts and does the comparison on those instead
    do_logsumexp: bool
        Whether to sum or logsumexp over seeds for LL metrics
    """
    assert metric in {'RMSE','MRE','MSE','poisson_NLL'}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if compare_daily_not_cumulative:
        y_true = get_daily_from_cumulative(y_true)
        y_pred = get_daily_from_cumulative(y_pred)
    else:
        assert metric not in ['poisson_NLL']

    if do_logsumexp:
        sum_or_logsumexp = logsumexp
    else:
        sum_or_logsumexp = np.sum

    if min_threshold is not None:
        orig_len = len(y_true)
        idxs = y_true >= min_threshold
        if not idxs.sum() > 0:
            print(y_true)
            print("Warning: NOT ENOUGH VALUES ABOVE THRESHOLD %s" % min_threshold)
            return np.nan
        y_true = y_true[idxs]
        y_pred = y_pred[:, idxs]
        num_dropped = orig_len - len(y_true)
        if num_dropped > 30:
            print('Warning: dropped %d dates after applying min_threshold %d' % (
                num_dropped, min_threshold))

    if metric == 'RMSE':
        return RMSE(y_true=y_true, y_pred=y_pred)
    elif metric == 'MRE':
        return MRE(y_true=y_true, y_pred=y_pred)
    elif metric == 'MSE':
        return MSE(y_true=y_true, y_pred=y_pred)
    elif metric == 'poisson_NLL':
        return poisson_NLL(
            y_true=y_true,
            y_pred=y_pred,
            sum_or_logsumexp=sum_or_logsumexp)

def evaluate_all_fitted_models_for_msa(msa_name, min_timestring=None,
                                        max_timestring=None,
                                        timestrings=None,
                                       required_properties=None,
                                       required_model_kwargs=None,
                                       recompute_losses=False,
                                       key_to_sort_by=None,
                                       old_directory=False):

    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs`
    """
     # NEW GB 20/05/2023 rebuttal
    # pd.set_option('max_columns', 50)
    # pd.set_option('display.width', 500)

    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_properties is None:
        required_properties = {}

    if timestrings is None:
        timestrings = filter_timestrings_for_properties(
            required_properties=required_properties,
            required_model_kwargs=required_model_kwargs,
            required_data_kwargs={'MSA_name':msa_name},
            min_timestring=min_timestring,
            max_timestring=max_timestring,
            old_directory=old_directory)
        print('Found %d fitted models for %s' % (len(timestrings), msa_name))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    if recompute_losses:
        nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)

    results = []
    start_time = time.time()
    for ts in timestrings:
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False,
            load_fast_results_only=(not recompute_losses), old_directory=old_directory)
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']
        assert data_kwargs['MSA_name'] == msa_name

        if recompute_losses:
            loss_dict,_ = compare_model_vs_real_num_cases(
                nyt_outcomes,
                model_kwargs['min_datetime'],
                model_results=model_results,
                make_plot=False)
            
            fast_to_load_results['loss_dict'] = loss_dict 

        results_for_ts = {'timestring':ts,
                         'data_kwargs':data_kwargs,
                         'model_kwargs':model_kwargs,
                         'results':model_results,
                         'experiment_to_run':experiment_to_run}

        if 'final infected fraction' in fast_to_load_results:
            results_for_ts['final infected fraction'] = fast_to_load_results['final infected fraction']

        for result_type in ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']:
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

    end_time = time.time()
    print('Time to load and score all models: %.3fs -> %.3fs per model' %
          (end_time-start_time, (end_time-start_time)/len(timestrings)))
    results = pd.DataFrame(results)

    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results

def evaluate_all_fitted_models_for_experiment(
    experiment_to_run,
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
        result_types = ['loss_dict', 'train_loss_dict', 'test_loss_dict']
        # ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']
    results = []
    start_time = time.time()
    for i, ts in enumerate(timestrings):
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False, load_fast_results_only=True, old_directory=old_directory)
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']

        results_for_ts = {'timestring':ts,
                         'data_kwargs':data_kwargs,
                         'model_kwargs':model_kwargs,
                         'results':model_results,
                         'experiment_to_run':experiment_to_run}

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

    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results

def test_if_kwargs_match(req_properties, req_data_kwargs,
                         req_model_kwargs, test_data_and_model_kwargs):
    # check whether direct properties in test_data_and_model_kwargs match
    prop_match = all([
        req_properties[key] == test_data_and_model_kwargs[key] 
        for key in req_properties if key not in ['data_kwargs', 'model_kwargs']])
    if not prop_match:
        return False

    # check whether data kwargs match
    test_data_kwargs = test_data_and_model_kwargs['data_kwargs']
    data_match = all([req_data_kwargs[key] == test_data_kwargs[key] for key in req_data_kwargs])
    if not data_match:
        return False

    # check if non-dictionary model kwargs match
    kwargs_keys = set([key for key in req_model_kwargs if 'kwargs' in key])
    test_model_kwargs = test_data_and_model_kwargs['model_kwargs']
    model_match = all([req_model_kwargs[key] == test_model_kwargs[key] for key in req_model_kwargs if key not in kwargs_keys])
    if not model_match:
        return False

    # check if elements within dictionary model kwargs match
    for kw_key in kwargs_keys:
        req_kwargs = req_model_kwargs[kw_key]
        test_kwargs = test_model_kwargs[kw_key]
        kw_match = all([req_kwargs[k] == test_kwargs[k] for k in req_kwargs])
        if not kw_match:
            return False
    return True    

def filter_timestrings_for_properties(required_properties=None,
                                      required_model_kwargs=None,
                                      required_data_kwargs=None,
                                      min_timestring=None,
                                      max_timestring=None,
                                      return_msa_names=False,
                                      old_directory=False):
    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs
    """
    if required_properties is None:
        required_properties = {}
    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_data_kwargs is None:
        required_data_kwargs = {}
    if max_timestring is None:
        max_timestring = str(
            datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
    print("Loading models with timestrings between %s and %s" % (str(min_timestring), max_timestring))
    if old_directory:
        config_dir = os.path.join(OLD_FITTED_MODEL_DIR, 'data_and_model_configs')
    else:
        config_dir = os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs')
    matched_timestrings = []
    msa_names = []
    configs_to_evaluate = os.listdir(config_dir)
    print("%i files in directory %s" % (len(configs_to_evaluate), config_dir))
    for fn in configs_to_evaluate:
        if fn.startswith('config_'):
            timestring = fn.lstrip('config_').rstrip('.pkl')
            if (timestring < max_timestring) and (min_timestring is None or timestring >= min_timestring):
                f = open(os.path.join(config_dir, fn), 'rb')
                data_and_model_kwargs = pickle.load(f)
                f.close()

                #print(data_and_model_kwargs)
                
                if test_if_kwargs_match(
                    required_properties,
                    required_data_kwargs,
                    required_model_kwargs,
                    data_and_model_kwargs):                    
                    matched_timestrings.append(timestring)
                    msa_names.append(data_and_model_kwargs['data_kwargs']['MSA_name'])

    if not return_msa_names:
        return matched_timestrings
    else:
        return matched_timestrings, msa_names

    return matched_timestrings

def load_model_and_data_from_timestring(
    timestring, verbose=False, load_original_data=False,
    load_full_model=False, load_fast_results_only=True,
    load_filtered_data_model_was_fitted_on=False,
    old_directory=False):

    if verbose:
        print("Loading model from timestring %s" % timestring)
    if old_directory:
        model_dir = OLD_FITTED_MODEL_DIR
    else:
        model_dir = FITTED_MODEL_DIR
    f = open(os.path.join(model_dir, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')
    data_and_model_kwargs = pickle.load(f)
    f.close()
    model = None
    model_results = None
    f = open(os.path.join(model_dir, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'rb')
    fast_to_load_results = pickle.load(f)
    f.close()

    if not load_fast_results_only:
        if SAVE_MODEL_RESULTS_SEPARATELY:
            f = open(os.path.join(helper.FITTED_MODEL_DIR, 'model_results', 'model_results_%s.pkl' % timestring), 'rb')
            model_results = pickle.load(f)
            f.close()

        if load_full_model:
            # filepath = os.path.join(model_dir, 'full_models', 'fitted_model_%s.pkl' % timestring), 'rb'
            # f = open(filepath)
            # f.close()
            filepath = os.path.join(model_dir, 'full_models', 'fitted_model_%s.gzip' % timestring)
            with gzip.open(filepath, "rb") as f:
                model = pickle.load(f)
            

    if load_original_data:
        if verbose:
            print("Loading original data as well...warning, this may take a while")
        d = helper.load_dataframe_for_individual_msa(**data_and_model_kwargs['data_kwargs'])
    else:
        d = None

    if load_filtered_data_model_was_fitted_on:
        # if true, return the data after all the filtering, along with the model prior to fitting.
        # data_kwargs = data_and_model_kwargs['data_kwargs'].copy()
        # model_kwargs = data_and_model_kwargs['model_kwargs'].copy()
        # model_kwargs['return_model_and_data_without_fitting'] = True
        # unfitted_model = fit_and_save_one_model(timestring=None,
        #                              model_kwargs=model_kwargs,
        #                              data_kwargs=data_kwargs,
        #                              train_test_partition=None)
        # filtered_data = unfitted_model.d
        # return model, data_and_model_kwargs, d, model_results, fast_to_load_results, filtered_data, unfitted_model
        print("Wrong turn! You made a mess with imports")
        return None
    else:
        return model, data_and_model_kwargs, d, model_results, fast_to_load_results
