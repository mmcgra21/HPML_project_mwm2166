import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import io

metric_aggregate = {
    'inst_executed': 'sum',
    'inst_per_warp': 'median',
    'sm_efficiency': 'median',
    'flop_count_hp': 'sum',
    'flop_count_sp': 'sum',
    'dram_read_bytes': 'sum',
    'dram_write_bytes': 'sum',
    'gld_transactions': 'sum',
    'gst_transactions': 'sum',
    'ipc': 'median',
    'achieved_occupancy': 'median'
}

datatypes = {
        'Device': object,
        'Context': int,
        'Stream': int,
        'Kernel': object
    }
metrics = []
dfs = dict()
for filename in os.listdir('profiles'):
    if 'profile_' not in filename:
        continue
    with open('profiles/' + filename, 'r') as f:
        lines = f.readlines()
    specifier = filename.split('_')[1]
    if specifier == 'f':
        optimizer = 'Fairseq'
    elif specifier == 'fa':
        optimizer = 'Fairseq+Apex'
    elif specifier == 'l':
        optimizer = 'LightSeq2'
    # Find line that contains "Profiling result:"
    for start_idx, line in enumerate(lines):
        if 'Profiling result:' in line:
            break
    names_idx = start_idx+1
    column_names = lines[names_idx].replace('\n','').split()
    data_start_idx = start_idx+3
    # Figure out where to split each line
    splits = [0]
    flag = False
    for idx,ch in enumerate(lines[names_idx].replace('\n','')):
        if flag and ch == ' ':
            splits.append(idx)
            flag = False
        elif ~flag and ch != ' ':
            flag = True

    # Split string using splits array
    data = []
    for line in lines[data_start_idx:]:
        l = []
        for idx, i in enumerate(splits):
            if idx == len(splits)-1:
                tmp = line.replace('\n','')[i:].strip()
            else:
                tmp = line.replace('\n','')[i:splits[idx+1]].strip()
            if column_names[idx] == 'dram_utilization' and tmp != '<INVALID>':
                tmp = tmp.split('(')[1].split(')')[0]
            l.append(tmp)
        data.append(l)

    df = pd.DataFrame(data, columns=column_names)
    df.replace(['<INVALID>'], np.nan, inplace=True)
    for name in column_names:
        if name not in datatypes:
            if name not in metrics:
                metrics.append(name)
            datatypes[name] = float
    df = df.astype(datatypes)
    dfs[optimizer] = df

metrics = list(metric_aggregate.keys())
horder = sorted(list(dfs.keys()))

df_all = pd.DataFrame()
for opt,df in dfs.items():
    df['Optimizer'] = opt
    df_all = pd.concat([df_all, df])
df_all_agg = pd.DataFrame()
names = []
for idx,metric in enumerate(metrics):
    name = f'{metric} ({metric_aggregate[metric]})'
    if idx == 0:
        tmp = df_all.groupby('Optimizer')[metric].agg([metric_aggregate[metric]])
        tmp[metric] = tmp[metric_aggregate[metric]]
        df_all_agg[name] = tmp[metric]
    else:
        tmp = df_all.groupby('Optimizer')[metric].agg([metric_aggregate[metric]])
        tmp[metric] = tmp[metric_aggregate[metric]]
        df_all_agg[name] = tmp[metric]
    names.append(name)

indices = list(df_all_agg.index)
for key in indices:
    if 'Fairseq' in key:
        break
df_tmp = df_all_agg.copy()
for idx in indices:
    df_all_agg.loc[idx] = df_tmp.loc[idx]/df_tmp.loc[key]

df_all_agg = df_all_agg.reset_index(inplace=False)
df_all_agg = pd.melt(df_all_agg, id_vars=['Optimizer'], value_vars=names)
df_all_agg['Metrics'] = df_all_agg['variable']
df_all_agg['Values'] = df_all_agg['value']
df_all_agg = df_all_agg.drop(columns=['variable','value'])

fig, ax = plt.subplots(figsize=(6, 4))
p = sns.barplot(x='Metrics',y='Values',hue='Optimizer',data=df_all_agg,ax=ax,hue_order=horder)
ax.set_xlabel('Metrics')
ax.set_ylabel(f'Values scaled w.r.t. {key}')
ax.tick_params(axis='x', rotation=90)
ax.legend(loc='upper left')
ax.get_legend().set_title("")

fig.savefig(f'plots/profiling_metrics.png', bbox_inches="tight")
