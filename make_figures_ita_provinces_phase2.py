#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from helper_methods_for_plotting import *


ref_df = pd.read_csv("italian_data/cap_comuni.csv")
provinces = ref_df[['nome_regione','nome_provincia']].drop_duplicates()
population_data = ref_df[['nome','popolazione','nome_provincia']]
muns = ref_df[['nome_regione','nome_provincia','nome']].drop_duplicates()


phase = 2
min_t = '2022_06_16_12'
ACCEPTABLE_LOSS_TOLERANCE = 1.15 #too many models in Lombardia

# ## Lombardia 
# Load and inspect model data:
msa = 'Lombardia'
subset = None
_, non_ablation_df, timestrings = get_models_df_experiment(
    phase=phase,region=msa,subset=subset,min_t=min_t)
model_df, table_df = get_model_summaries(non_ablation_df, [msa])
fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
min_timestring, max_timestring = timestrings
plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
thing_to_plot='cases'
axes = plot_model_results_msa(
    msa, axes, non_ablation_df, plotdate, 
    subset=subset, thing_to_plot=thing_to_plot, 
    threshold=ACCEPTABLE_LOSS_TOLERANCE)
fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
plt.subplots_adjust(hspace=.1)
plt.tight_layout()
prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")

# ## Lombardia - Milano
# Load and inspect model data:
msa = 'Lombardia'
muns_r = muns[muns['nome_regione']==msa]
prov_r = muns_r['nome_provincia'].unique()    
for subset in prov_r:
    _, non_ablation_df, timestrings = get_models_df_experiment(
        phase=phase,region=msa,subset=subset,min_t=min_t)
    model_df, table_df = get_model_summaries(non_ablation_df, [msa])
    fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
    min_timestring, max_timestring = timestrings
    plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    thing_to_plot='cases'
    axes = plot_model_results_msa(
        msa, axes, non_ablation_df, plotdate, 
        subset=subset, thing_to_plot=thing_to_plot, 
        threshold=ACCEPTABLE_LOSS_TOLERANCE)

    fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
    plt.subplots_adjust(hspace=.1)
    plt.tight_layout()
    prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
    fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
    table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")

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


# ## Lazio 
ACCEPTABLE_LOSS_TOLERANCE = 1.2

msa = 'Lazio'
subset=None
_, non_ablation_df, timestrings = get_models_df_experiment(
    phase=phase,region=msa, min_t=min_t, subset=subset)
model_df, table_df = get_model_summaries(
    non_ablation_df, [msa],
    threshold=ACCEPTABLE_LOSS_TOLERANCE
)
fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
min_timestring, max_timestring = timestrings
plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
thing_to_plot='cases'
axes = plot_model_results_msa(
    msa, axes, non_ablation_df, plotdate, 
    subset=subset, thing_to_plot=thing_to_plot, 
    threshold=ACCEPTABLE_LOSS_TOLERANCE)
fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
plt.subplots_adjust(hspace=.1)
plt.tight_layout()
prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")


# ## Lazio - Roma
msa = 'Lazio'
muns_r = muns[muns['nome_regione']==msa]
prov_r = muns_r['nome_provincia'].unique()    
for subset in prov_r:
    _, non_ablation_df, timestrings = get_models_df_experiment(
        phase=phase,region=msa,subset=subset,min_t=min_t)
    model_df, table_df = get_model_summaries(non_ablation_df, [msa])
    fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
    min_timestring, max_timestring = timestrings
    plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    thing_to_plot='cases'
    axes = plot_model_results_msa(
        msa, axes, non_ablation_df, plotdate, 
        subset=subset, thing_to_plot=thing_to_plot, 
        threshold=ACCEPTABLE_LOSS_TOLERANCE)

    fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
    plt.subplots_adjust(hspace=.1)
    plt.tight_layout()
    prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
    fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
    table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")

# ## Lazio - Latina
# msa = 'Lazio'
# subset = 'Latina'
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

# ## Campania
msa = 'Campania'
subset=None
_, non_ablation_df, timestrings = get_models_df_experiment(
    phase=phase,region=msa, min_t=min_t, subset=subset)
model_df, table_df = get_model_summaries(
    non_ablation_df, [msa],
    threshold=ACCEPTABLE_LOSS_TOLERANCE)
fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
min_timestring, max_timestring = timestrings
plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
thing_to_plot='cases'
axes = plot_model_results_msa(
    msa, axes, non_ablation_df, plotdate, 
    subset=subset, thing_to_plot=thing_to_plot, 
    threshold=ACCEPTABLE_LOSS_TOLERANCE)
fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
plt.subplots_adjust(hspace=.1)
plt.tight_layout()
prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")


# ## Campania - Napoli
msa = 'Campania'
muns_r = muns[muns['nome_regione']==msa]
prov_r = muns_r['nome_provincia'].unique()    
for subset in prov_r:
    _, non_ablation_df, timestrings = get_models_df_experiment(
        phase=phase,region=msa,subset=subset,min_t=min_t)
    model_df, table_df = get_model_summaries(non_ablation_df, [msa])
    fig, axes = plt.subplots(1, 2, figsize=(14,7), sharex=True, sharey=True)
    min_timestring, max_timestring = timestrings
    plotdate = f"{reformat_date(min_timestring)}_{reformat_date(max_timestring)}"
    thing_to_plot='cases'
    axes = plot_model_results_msa(
        msa, axes, non_ablation_df, plotdate, 
        subset=subset, thing_to_plot=thing_to_plot, 
        threshold=ACCEPTABLE_LOSS_TOLERANCE)

    fig.suptitle(t=f"{msa} - {subset}", fontsize=20, x=.55)
    plt.subplots_adjust(hspace=.1)
    plt.tight_layout()
    prefix_to_save_plot_with = f'trajectory_{thing_to_plot}_oos_vs_full_fit_{msa}_{subset}_{plotdate}'
    fig.savefig(f"plots/model_fit/{prefix_to_save_plot_with}.png", dpi=300, bbox_inches='tight')
    table_df.to_csv(f"plots/model_fit/{prefix_to_save_plot_with}.csv")

# ## Campania - Salerno
# msa = 'Campania'
# subset = 'Salerno'
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






