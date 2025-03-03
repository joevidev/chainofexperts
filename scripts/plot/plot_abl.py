import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

fig, ax = plt.subplots(figsize=(12*0.7, 7*0.7))
data = pd.read_csv('data5.tsv', sep='\t')

data.columns = [col.strip() for col in data.columns]

steps = np.arange(10, (len(data) + 1) * 10, 10)

methods = [
    'dsv2_coe_3-ab_64ept-8tpk-2itr-noig - val/loss', 
    'dsv2_coe_3-ab_64ept-8tpk-2itr-ore - val/loss',  
    'dsv2_coe_3-cc_64ept-8tpk-2itr - val/loss',    
    'dsv2_coe_3-fl_64ept-8tpk-1itr - val/loss'     
]

labels = [
    'CoE w/o Independent Gate',
    'CoE w/o Inner Residual',
    'CoE-2(4/64)',
    'MoE(8/64)'
]

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
markers = ['o', 's', 'D', '^']
linestyles = ['-', '--', ':', '-.']

for i, (method, label) in enumerate(zip(methods, labels)):
    if method in data.columns:
        ax.loglog(steps, data[method].values, 
                 label=label, 
                 linestyle=linestyles[i], 
                 linewidth=2.5,
                 marker=markers[i],
                 markersize=6,
                 markevery=10,
                 color=colors[i])

ax.set_xlabel('Steps', fontweight='bold', fontsize=14)
ax.set_ylabel('Validation Loss (log scale)', fontweight='bold', fontsize=14)
ax.set_title('Ablation Study: Independent Gate & Inner Residual', fontweight='bold', fontsize=16)
ax.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
ax.grid(True, which='major', linestyle='-', alpha=0.5)

ax.set_xlim(370, 1000)
ax.set_ylim(1, 2)

ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('plot_abl.png', dpi=300, bbox_inches='tight')
plt.show()