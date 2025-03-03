import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Set better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = '#333333'

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18*0.79, 7*0.79), gridspec_kw={'width_ratios': [1.1, 1]})

# ================ LEFT SUBPLOT - CONVERGENCE PERFORMANCE ================
# Load the data
data = pd.read_csv('data3.tsv', sep='\t')

# Extract steps and values
steps = data['Step'].values
# For the left plot, we'll focus on the models mentioned in your requirements
# CoE-2(4/48) and MoE(8/64)
# Assuming these are in your columns, adjust as needed
left_columns = ['64ept-8tpk-1itr', '64ept-4tpk-2itr', '48ept-4tpk-2itr']
left_labels = ['MoE(8/64)', 'CoE-2(4/64)', 'CoE-2(4/48)']

# Find the index where steps >= 100
start_idx = np.where(steps >= 100)[0]
start_idx = 0 if len(start_idx) == 0 else start_idx[0]

# Define colors and markers for left plot
left_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
left_markers = ['o', 's', 'D']
left_linestyles = ['-', '--', ':']

# Create plot with log scales for selected columns
for i, (column, label) in enumerate(zip(left_columns, left_labels)):
    if column in data.columns:
        ax1.loglog(steps[start_idx:], data[column].values[start_idx:], 
                  label=label, 
                  linestyle=left_linestyles[i], 
                  linewidth=2.5,
                  marker=left_markers[i],
                  markersize=5,
                  markevery=5,
                  color=left_colors[i])

# Add labels and title for left plot
ax1.set_xlabel('Steps (log scale)', fontweight='bold', fontsize=14)
ax1.set_ylabel('Validation Loss (log scale)', fontweight='bold', fontsize=14)
ax1.set_title('Performance Comparison', fontweight='bold', fontsize=16)
ax1.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax1.grid(True, which='minor', linestyle=':', alpha=0.4)
ax1.grid(True, which='major', linestyle='-', alpha=0.5)
ax1.set_xlim(400, 1000)
ax1.set_ylim(1, 2)
ax1.set_facecolor('#f8f9fa')

# ================ RIGHT SUBPLOT - EFFICIENCY METRICS ================
# Create data for the right plot - Resource Efficiency
models = ['CoE-2(4/48)', 'MoE(8/64)', 'CoE-2(4/64)']
params = [412.63, 544.51, 544.75]  # in million parameters
memory = [9.64, 11.70, 11.70]  # in GB

# Normalize the values relative to MoE(8/64)
params_norm = [p/params[1]*100 for p in params]
memory_norm = [m/memory[1]*100 for m in memory]

# Set width of bars
barWidth = 0.25

# Set positions of the bars on X axis
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]

# Create grouped bars
ax2.bar(r1, params_norm, width=barWidth, label='Parameters', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1)
ax2.bar(r2, memory_norm, width=barWidth, label='Memory', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)

# Show the reference line at 100%
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Add values on the bars
for i, (p, m) in enumerate(zip(params_norm, memory_norm)):
    ax2.text(r1[i], p+1, f"{p:.1f}%", ha='center', va='bottom', fontsize=10)
    ax2.text(r2[i], m+1, f"{m:.1f}%", ha='center', va='bottom', fontsize=10)

# Highlight the memory reduction of CoE-2(4/48)
memory_reduction = 100 - memory_norm[0]
ax2.annotate(f"-{memory_reduction:.1f}% memory", 
            xy=(r2[0], memory_norm[0]+20), 
            xytext=(r2[0]-0.1, memory_norm[0]+10),
            fontsize=11,
            weight='bold',
            color='#2ca02c',
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc="#2ca02c", ec="none"))

# Add xticks for each model group
ax2.set_xticks([r + barWidth for r in range(len(models))])
ax2.set_xticklabels(models)

# Set labels and title for right plot
ax2.set_ylabel('Relative to MoE(8/64) (%)', fontweight='bold', fontsize=14)
ax2.set_title('Resource Efficiency', fontweight='bold', fontsize=16)
ax2.legend(frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper right')
ax2.set_ylim(0, 130)
ax2.set_facecolor('#f8f9fa')

# Add gridlines
ax2.grid(axis='y', linestyle='-', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Main title for the entire figure
plt.suptitle('CoE Reduces Memory Requirements while Maintaining Performance', 
            fontweight='bold', fontsize=18, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plot_efficiency.png', dpi=300, bbox_inches='tight')
plt.show()