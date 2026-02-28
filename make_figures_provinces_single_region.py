#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from helper_methods_for_plotting import *
import argparse
import matplotlib.dates as mdates

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str)
    parser.add_argument('--phase', type=int, default=2)
    parser.add_argument('--min_t', type=str, default='2022_06_20')
    parser.add_argument('--max_t', type=str, default='2022_06_23')
    args = parser.parse_args()    

    phase = args.phase   #2
    min_t = args.min_t   #'2022_06_20'
    max_t = args.max_t   #'2022_06_20'
    msa = args.region    #'Lombardia'
    
    ref_df = pd.read_csv("italian_data/cap_comuni.csv")
    provinces = ref_df[['nome_regione','nome_provincia']].drop_duplicates()
    population_data = ref_df[['nome','popolazione','nome_provincia']]
    muns = ref_df[['nome_regione','nome_provincia','nome']].drop_duplicates()
    
#     if msa == 'Lombardia':
#         ACCEPTABLE_LOSS_TOLERANCE = 1.15 #too many models in Lombardia
#     else:
    
    ACCEPTABLE_LOSS_TOLERANCE = 1.2

    # ## Lombardia 
    # Load and inspect model data:    
    subset = None
    _, non_ablation_df, timestrings = get_models_df_experiment(
        phase=phase,region=msa,subset=subset,min_t=min_t,max_t=max_t)
    model_df, table_df = get_model_summaries(non_ablation_df, [msa])
    # plot data
    fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
    min_timestring, max_timestring = timestrings
    plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    thing_to_plot='cases'
    
    axes, all_new_data = plot_model_results_msa(
        msa, axes, non_ablation_df, plotdate, 
        subset=subset, thing_to_plot=thing_to_plot, 
        threshold=ACCEPTABLE_LOSS_TOLERANCE)
    
    data_for_testing_full = get_prediction_data_for_testing(all_new_data[0])
    data_for_testing_ool = get_prediction_data_for_testing(all_new_data[1])
    
    for ax in axes:
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%b-%d'))
    
    fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
    plt.subplots_adjust(hspace=.1)
    plt.tight_layout()
    prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
    fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.pdf", dpi=300, bbox_inches='tight')
    table_df.to_csv(f"plots/model_fit/tables/{prefix_to_save_plot_with}.csv")
    
    data_for_testing_full.to_csv(
        f"plots/model_fit/tables/data_for_testing_full_{msa}_{subset}.csv", 
        index=False)
    data_for_testing_ool.to_csv(
        f"plots/model_fit/tables/data_for_testing_ool_{msa}_{subset}.csv", 
        index=False)

    # ## Lombardia - Milano
    # Load and inspect model data:
    muns_r = muns[muns['nome_regione']==msa]
    prov_r = list(muns_r['nome_provincia'].unique())
    
    short_provs = {
        'Lombardia': ["Milano","Brescia"],
        'Lazio': ["Roma","Latina"], 
        'Campania': ["Napoli","Salerno"]}    
    
    assert all([p in prov_r for p in short_provs[msa]])    
    
    prov_r = short_provs[msa]
    
    for subset in prov_r:
        _, non_ablation_df, timestrings = get_models_df_experiment(
            phase=phase,region=msa,subset=subset,min_t=min_t,max_t=max_t)
        model_df, table_df = get_model_summaries(non_ablation_df, [msa])
        fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
        min_timestring, max_timestring = timestrings
        plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
        thing_to_plot='cases'
        
        axes, all_new_data = plot_model_results_msa(
            msa, axes, non_ablation_df, plotdate, 
            subset=subset, thing_to_plot=thing_to_plot, 
            threshold=ACCEPTABLE_LOSS_TOLERANCE)
        
        # TODO: 0 is oos, 1 is full
        data_for_testing_full = get_prediction_data_for_testing(all_new_data[0])
        data_for_testing_ool = get_prediction_data_for_testing(all_new_data[1])

        for ax in axes:
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%b-%d'))
        
        fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
        plt.subplots_adjust(hspace=.1)
        plt.tight_layout()
        prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
        fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.pdf", dpi=300, bbox_inches='tight')
        table_df.to_csv(f"plots/model_fit/tables/{prefix_to_save_plot_with}.csv")
        
        data_for_testing_full.to_csv(
            f"plots/model_fit/tables/data_for_testing_full_{msa}_{subset}.csv", 
            index=False)
        data_for_testing_ool.to_csv(
            f"plots/model_fit/tables/data_for_testing_ool_{msa}_{subset}.csv", 
            index=False)
        
        
    # ## Lombardia - Bergamo
    # msa = 'Lombardia'
    # subset = 'Bergamo'
    # _, non_ablation_df, timestrings = get_models_df_experiment(
    #     phase=phase,region=msa,subset=subset,min_t=min_t)
    # model_df, table_df = get_model_summaries(non_ablation_df, [msa])
    # fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
    # min_timestring, max_timestring = timestrings
    # plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    # thing_to_plot='cases'
    # axes = plot_model_results_msa(
    #     msa, axes, non_ablation_df, plotdate, 
    #     subset=subset, thing_to_plot=thing_to_plot, 
    #     threshold=ACCEPTABLE_LOSS_TOLERANCE)
    # fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
    # plt.subplots_adjust(hspace=.1)
    # plt.tight_layout()
    # prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
    # fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
    # table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")


