import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os 
import datetime 
import time
import copy 
import argparse

from covid_constants_and_util import PATH_TO_SAVED_CHARACTERISTICS, HIGHLIGHT_MSA, MIN_DATETIME, MAX_DATETIME, TRAIN_TEST_PARTITION
from model_evaluation import evaluate_all_fitted_models_for_experiment, load_model_and_data_from_timestring
from utilities import apply_smoothing
from textwrap import wrap
import helper_methods_for_aggregate_data_analysis as helper
import matplotlib.dates as mdates
# from matplotlib import ticker as tick

from search_model_results import get_experiments_results
from helper_methods_for_aggregate_data_analysis import load_dataframe_for_individual_province

def get_metadata_filename(msa_name, min_datetime, max_datetime, clip_visits=True, correct_visits=True):
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
    filename = os.path.join(PATH_TO_SAVED_CHARACTERISTICS, 'metadata_%s.pkl' % fn)
    return filename

def get_frac_infected_over_time_per_category(
    results_df, poi_and_cbg_characteristics, msa_name, categories_to_plot, subset='all', var_string='POI_categories'):
    msa_df = results_df[results_df['MSA_name'] == msa_name]
    poi_categories = poi_and_cbg_characteristics[msa_name][var_string]
    pretty_names = np.array([
        SUBCATEGORIES_TO_PRETTY_NAMES[x] 
        if x in SUBCATEGORIES_TO_PRETTY_NAMES else x for x in poi_categories])    
    results_per_seed = []
    
    for ts in msa_df.timestring.values:
        print(ts)
        model, _, _, _, fast_to_load_results = load_model_and_data_from_timestring(
            ts, load_fast_results_only=False, load_full_model=True)
        num_cases_per_poi = model.history[subset]['num_cases_per_poi']
        pop_size = model.history[subset]['total_pop']
        assert len(num_cases_per_poi.shape) == 3  # must be seed x poi x time
        for s in range(len(num_cases_per_poi)):  # iterate through seeds
            total_frac_infected_at_pois_per_day = np.sum(num_cases_per_poi[s], axis=0) / pop_size  # sum over all pois
            frac_infected_per_cat_and_day = []
            for cat in categories_to_plot:
                cat_idx = pretty_names == cat
                assert np.sum(cat_idx) >= 10  # there should be at least 10 POIs in this category
                frac_infected_at_cat_per_day = np.sum(num_cases_per_poi[s][cat_idx], axis=0) / pop_size  # sum over pois within cat
                frac_infected_per_cat_and_day.append(frac_infected_at_cat_per_day)
            results_per_seed.append((frac_infected_per_cat_and_day, total_frac_infected_at_pois_per_day))
    print('Num params * seeds:', len(results_per_seed))
    return results_per_seed

def moving_average(x, w):
    # see: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--contribution', 
        help='Plot contributions according to which dimension',
        default='POI_provinces')
    parser.add_argument(
        '--msa', 
        help='Select specific Region to plot ',
        default=None)        
    parser.add_argument(
        '--min_timestring_best_fit', 
        type=str, 
        help='Min timestring for rerun_best_models',
        default='2022_09_28')     
    args = parser.parse_args()

    ## Scripts
    ateco_code = pd.read_csv("italian_data/MCC_to_ATECO_final.csv", encoding='latin')
    ateco_code_ref = ateco_code[['ateco_code','ateco_description']]
    ateco_code_ref = ateco_code_ref.dropna().drop_duplicates()
    ateco_code_ref = ateco_code_ref.set_index('ateco_code')['ateco_description'].to_dict()
    top_categories = [
        470,610,350,860,510,560,660,
        790,450,640,850,490,650,550,520,920]
    pretty_top_categories = [
        ateco_code_ref[cat] if cat in ateco_code_ref else cat 
        for cat in top_categories]

    SUBCATEGORIES_TO_PRETTY_NAMES = {str(int(k)):v for k,v in ateco_code_ref.items()}

    ateco_description = pd.read_csv("italian_data/ateco_simplified_translated.csv")
    ateco_code_ref = ateco_description.set_index('ateco_code')['Titolo NACE Rev. 2'].to_dict()
    #ateco_ref[-1] = "Other"

    top_categories = [
        470,610,350,860,510,560,660,
        790,450,640,850,490,650,550,520,920]
    pretty_top_categories = [
        ateco_code_ref[cat] if cat in ateco_code_ref else cat 
        for cat in top_categories]

    SUBCATEGORIES_TO_PRETTY_NAMES = {str(int(k)):v for k,v in ateco_code_ref.items()}

    PATH_TO_SAVED_CHARACTERISTICS = os.path.join(
        'data', 'all_aggregate_data',
        'poi_metadata',
        'filter5_rs', 'second')

    # Lombardia
    ## General set-up
    if args.msa is None:
        all_msas = ['Lazio','Campania','Lombardia']
    else:
        all_msas = [args.msa]
        
    for msa_name in all_msas:
        BIGGEST_MSAS = [msa_name]
        HIGHLIGHT_MSA = msa_name 
        
        print(f"Exporting figure for region {HIGHLIGHT_MSA}")
        
        # reloads poi_and_cbg_characteristics
        p = get_metadata_filename(HIGHLIGHT_MSA,MIN_DATETIME, MAX_DATETIME)
        assert os.path.exists(p)
        print("last modified: %s" % time.ctime(os.path.getmtime(p)))

        with open(p, 'rb') as file_handle:
            poi_and_cbg_characteristics = pickle.load(file_handle)
        
        for k,v in poi_and_cbg_characteristics[HIGHLIGHT_MSA].items():
            print(k, len(v))

        ## Extended Data Analyses 

        ### Distribution of POI infections over time
        res = get_experiments_results(
            experiment = 'rerun_best_models_and_save_cases_per_poi',
            phase=2,
            region=msa_name,
            min_t=args.min_timestring_best_fit)
        min_timestring = res[0]
        pt1 = "_".join(res[-1].split("_")[:4])
        pt2 = "_"+str(int(res[-1].split("_")[4])+2)
        max_timestring = pt1+pt2

        models_df = evaluate_all_fitted_models_for_experiment(
            'rerun_best_models_and_save_cases_per_poi', 
            min_timestring=min_timestring, 
            max_timestring=max_timestring
        )
        models_df['MSA_name'] = models_df['data_kwargs'].map(lambda x:x['MSA_name'])
        models_df['original_timestring'] = models_df['model_kwargs'].map(
            lambda x:x['model_quality_dict']['model_timestring'])
        models_df['how_to_select_best_grid_search_models'] = models_df['model_kwargs'].map(
            lambda x:x['model_quality_dict']['how_to_select_best_grid_search_models'])

        population_data_msa = load_dataframe_for_individual_province(HIGHLIGHT_MSA)
        a = set(poi_and_cbg_characteristics[msa_name]['POI_municipalities'])
        b = set(population_data_msa['municipality'].apply(str.lower)) 
        print(f"% missing municipalities {len(a.difference(b))/len(a)}")
        
        population_data_msa = population_data_msa.set_index('municipality')
        province_mapping = {}
        for c in population_data_msa.columns:
            if c not in ['population','track']:
                prov_data = population_data_msa[[c]]
                prov_data = prov_data[prov_data[c]==1]
                for m in prov_data.index:
                    province_mapping[m.lower()] = c
        poi_and_cbg_characteristics[msa_name]['POI_provinces'] = [
            province_mapping.get(m, 'Other')
            for m in poi_and_cbg_characteristics[msa_name]['POI_municipalities']
        ]

        poi_categories = poi_and_cbg_characteristics[msa_name][args.contribution]
        pretty_names = np.array([
            SUBCATEGORIES_TO_PRETTY_NAMES[x] 
            if x in SUBCATEGORIES_TO_PRETTY_NAMES else x for x in poi_categories])    
        unique, counts = np.unique(pretty_names, return_counts=True)
        pretty_top_categories = pd.Series(dict(zip(unique, counts))).sort_values(ascending=False)
        
        if args.contribution == 'POI_provinces':
            pretty_top_categories = pretty_top_categories.index.tolist()
        else:
            pretty_top_categories = pretty_top_categories[pretty_top_categories>=10].index.tolist()

        msa_results_over_seeds = {}
        for msa in BIGGEST_MSAS:
            print(msa)
            msa_results_over_seeds[msa] = get_frac_infected_over_time_per_category(
                models_df, poi_and_cbg_characteristics, 
                msa, pretty_top_categories, 
                var_string=args.contribution)

        # get the categories that contribute the most to infections over time
        all_avg_props = []
        all_min_props = []
        for i, cat in enumerate(pretty_top_categories):
            keep_cat = True
            avg_prop_per_msas = []
            for msa in msa_results_over_seeds.keys():
                results_per_seed = msa_results_over_seeds[msa]
                prop_infections_over_seeds = []
                for s in range(len(results_per_seed)):
                    frac_infected = results_per_seed[s][0][i]  # fraction of population infected at this cat per day
                    total_frac_infected = results_per_seed[s][1]  # total fraction of population infected at POIs per day
                    prop_infections = np.sum(frac_infected) / np.sum(total_frac_infected)  # proportion of all POI infections
                    prop_infections = np.nan_to_num(prop_infections)
                    prop_infections_over_seeds.append(prop_infections)
                avg_prop = np.mean(prop_infections_over_seeds)  # avg proportion over seeds
                avg_prop_per_msas.append(avg_prop)
            all_avg_props.append(np.mean(avg_prop_per_msas))
            all_min_props.append(np.min(avg_prop_per_msas))

        order = np.argsort(-1 * np.array(all_avg_props))
        categories_to_plot = []
        if args.contribution == 'POI_provinces':
            for i in order:
                cat, avg_prop, min_prop = pretty_top_categories[i], all_avg_props[i], all_min_props[i]                
                categories_to_plot.append(cat)        
                if avg_prop > 0.0002:
                    print('%s: mean over MSAs = %.2f%%, min over MSAs = %.2f%%' % (cat, 100.* avg_prop, 100.* min_prop))                
        else:
            categories_to_plot = [
                'Retail trade, except of motor vehicles and motorcycles',
                'Electricity, gas, steam and air conditioning supply',
                'Warehousing and support activities for transportation',
                'Food and beverage service activities',
                'Human health activities',
                'Telecommunications',
                'Other personal service activities',
                'Motion picture, video and television programme production, sound recording and music publishing activities',
                'Financial service activities, except insurance and pension funding',
                'Veterinary activities',
                'Wholesale and retail trade and repair of motor vehicles and motorcycles',
                'Advertising and market research',
                'Land transport and transport via pipelines',
                'Activities auxiliary to financial services and insurance activities',]
            
            for i in order:
                cat, avg_prop, min_prop = pretty_top_categories[i], all_avg_props[i], all_min_props[i]                                
                if (avg_prop > 0.0002) & (cat not in categories_to_plot):
                    print('%s: mean over MSAs = %.2f%%, min over MSAs = %.2f%%' % (cat, 100.* avg_prop, 100.* min_prop))                
                    categories_to_plot.append(cat)        


        print('Final list of cats to plot', categories_to_plot)

        results_lombardia = copy.copy(msa_results_over_seeds[HIGHLIGHT_MSA])
        
        fig_region, ax = plt.subplots(1, 1, figsize=(10, 10))
        categories_in_results = pretty_top_categories 
        y_max=1
        smooth=True
        results_per_seed = results_lombardia

        dates = helper.list_datetimes_in_range(MIN_DATETIME, MAX_DATETIME)
        bottom = np.zeros(len(dates))
        colors = sns.color_palette(
            palette='inferno_r', 
            n_colors=len(categories_to_plot)+1)

        if args.contribution == 'POI_provinces':
            categories_to_plot.remove("Other")
            categories_in_results.remove("Other")

        for j,cat in enumerate(categories_to_plot):
            assert cat in categories_in_results
            i = categories_in_results.index(cat)

            prop_infections_over_seeds = []
            for s in range(len(results_per_seed)):
                frac_infected = results_per_seed[s][0][i]  # fraction of population infected at this cat per day
                total_frac_infected = results_per_seed[s][1]  # total fraction of population infected at POIs per day
                prop_infections = frac_infected / total_frac_infected  # proportion of POI infections at this cat per day
                prop_infections = np.nan_to_num(prop_infections)
                prop_infections_over_seeds.append(prop_infections)

            mean_prop_infections_per_day = np.mean(np.array(prop_infections_over_seeds), axis=0)  # average over seeds
            if smooth:                
                mean_prop_infections_per_day = apply_smoothing(mean_prop_infections_per_day, before=7, after=7)
            top = bottom + mean_prop_infections_per_day
            ax.fill_between(dates, bottom, top, label=cat,color=colors[j], alpha=.9)
            bottom = top
        ceiling = np.ones(len(bottom)) * y_max
        #assert all(bottom < ceiling)
        ax.fill_between(dates, bottom, ceiling, label='Other', color=colors[j+1], alpha=.9)
        
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(mdates.SU, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.set_xlabel('Date', fontsize=16)
        ylabel = 'Proportion of daily\nPOI infections'
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title(msa_name, fontsize=20)

        handles, labels = ax.get_legend_handles_labels()
        labels = [ '\n'.join(wrap(l, 50)) for l in labels]
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.4, 1), fontsize=14)

        ax.set_xlim(dates[0],dates[-1])
        ax.set_xlabel(None)

        if args.contribution != 'POI_provinces':
            ax.set_ylim(.85,1)  
            ax.vlines(TRAIN_TEST_PARTITION, ymin=.85,ymax=1, color='k', linewidth=1)  
        else:
            ax.set_ylim(0,1)    
            ax.vlines(TRAIN_TEST_PARTITION, ymin=0,ymax=1, color='k', linewidth=1)

        fig_region.savefig(
            f"plots/contributions/{args.contribution}_contribution_infections_{HIGHLIGHT_MSA}.pdf", 
            bbox_inches='tight',dpi=300)
        fig_region.savefig(
            f"plots/contributions/{args.contribution}_contribution_infections_{HIGHLIGHT_MSA}.png", 
            bbox_inches='tight',dpi=300)

        res = {}
        res_daily = {}
        for j,cat in enumerate(categories_to_plot):
            assert cat in categories_in_results
            i = categories_in_results.index(cat)

            prop_infections_over_seeds = []
            for s in range(len(results_per_seed)):
                frac_infected = results_per_seed[s][0][i]  # fraction of population infected at this cat per day
                total_frac_infected = results_per_seed[s][1]  # total fraction of population infected at POIs per day
                prop_infections = frac_infected / total_frac_infected  # proportion of POI infections at this cat per day
                prop_infections = np.nan_to_num(prop_infections)
                prop_infections_over_seeds.append(prop_infections)

            mean_prop_infections_per_day = np.mean(np.array(prop_infections_over_seeds), axis=0)  # average over seeds
            if smooth:
                mean_prop_infections_per_day = apply_smoothing(mean_prop_infections_per_day, before=7, after=7)    

            first = mean_prop_infections_per_day[0]
            last = mean_prop_infections_per_day[-1] 
            res[cat] = {'first':first, 'last':last}
            res_daily[cat] = mean_prop_infections_per_day

        res_df = pd.DataFrame(res).T
        res_df['growth'] = res_df['last'].sub(res_df['first']).div(res_df['first'])
        res_df.to_csv(f"plots/contributions/tables/{args.contribution}_contribution_infections_{HIGHLIGHT_MSA}.csv")
        
        res_daily_df = pd.DataFrame(res_daily)
        res_daily_df.to_csv(f"plots/contributions/tables/{args.contribution}_contribution_infections_{HIGHLIGHT_MSA}_daily.csv")