#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from helper_methods_for_plotting import *
import matplotlib.dates as mdates
from covid_constants_and_util import TRAIN_TEST_PARTITION
import argparse

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2)
    parser.add_argument('--min_t', type=str, default='2022_06_20')
    args = parser.parse_args()    

    phase = args.phase   #2
    min_t = args.min_t   #'2022_06_20'
    subset=None

    if phase==2:
        ACCEPTABLE_LOSS_TOLERANCE = 1.15
    else:
        ACCEPTABLE_LOSS_TOLERANCE = 1.2
    msa = 'Lombardia'
    _, non_ablation_df_lombardia, timestrings = get_models_df_experiment(
        phase=phase,region=msa, min_t=min_t, subset=subset)
    model_df, table_df = get_model_summaries(
        non_ablation_df_lombardia, [msa],
        threshold=ACCEPTABLE_LOSS_TOLERANCE)

    ACCEPTABLE_LOSS_TOLERANCE = 1.2
    msa = 'Lazio'
    _, non_ablation_df_lazio, timestrings = get_models_df_experiment(
        phase=phase,region=msa, min_t=min_t, subset=subset)
    model_df, table_df = get_model_summaries(
        non_ablation_df_lazio, [msa],
        threshold=ACCEPTABLE_LOSS_TOLERANCE)

    msa = 'Campania'
    _, non_ablation_df_campania, timestrings = get_models_df_experiment(
        phase=phase,region=msa, min_t=min_t, subset=subset)
    model_df, table_df_lazio = get_model_summaries(
        non_ablation_df_campania, [msa],
        threshold=ACCEPTABLE_LOSS_TOLERANCE)


    fig1, axes1 = plt.subplots(3, 1, figsize=(10,24))
    fig2, axes2 = plt.subplots(3, 1, figsize=(10,24))

    min_timestring, max_timestring = timestrings
    plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    thing_to_plot='cases'

    msas = ['Lombardia','Lazio','Campania']
    tols = [1.15,1.2,1.2]
    dfs = [non_ablation_df_lombardia,non_ablation_df_lazio,non_ablation_df_campania]

    for ax1,ax2,msa,tol,non_ablation_df in zip(axes1,axes2,msas,tols,dfs):
        axes = ax1,ax2
        axes = plot_model_results_msa(
            msa, axes, non_ablation_df, plotdate, 
            thing_to_plot=thing_to_plot, 
            threshold=tol)
        ax2.vlines(
            TRAIN_TEST_PARTITION, 
            ax2.get_ylim()[0],ax2.get_ylim()[1], 
            color='black', linestyle='dashed')
        
        for ax in [ax1,ax2]:
            ax.set_title(f"{msa}", fontsize=20)        
            ax.tick_params(axis='x', rotation=0)
            ax.set_ylabel('Daily infections', fontsize=18)
            ax.tick_params(labelsize=16)        
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter('%b-%d'))
        
#         ax2.xaxis.set_major_locator(
#             mdates.WeekdayLocator(byweekday=mdates.SU, interval=2))
#         ax2.xaxis.set_major_formatter(
#             mdates.DateFormatter('%b-%d'))        
#         ax2.set_title(f"{msa}", fontsize=20)
#         ax2.tick_params(axis='x', rotation=0)        
#         ax2.set_ylabel('Daily infections', fontsize=18)        
#         ax2.tick_params(labelsize=16)
    
#     plt.subplots_adjust(hspace=.1)
#     plt.tight_layout()

    fig1.savefig(f"plots/model_fit/three_regions_oo_fit_phase_{phase}.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"plots/model_fit/three_regions_full_fit_phase_{phase}.png", dpi=300, bbox_inches='tight')
    fig1.savefig(f"plots/model_fit/three_regions_oo_fit_phase_{phase}.pdf", dpi=300, bbox_inches='tight')
    fig2.savefig(f"plots/model_fit/three_regions_full_fit_phase_{phase}.pdf", dpi=300, bbox_inches='tight')

