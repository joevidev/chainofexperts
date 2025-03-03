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
# Replace this with your actual data reading code
# For demonstration, I'll create a template
steps = np.array([i*10 for i in range(1, 101)])  # 10 to 1000 by 10s
data = pd.read_csv('data4.tsv', sep='\t')

# Create synthetic data representing your convergence curves
# This should be replaced with your actual data
moe_8_64 = data['64ept-8tpk-1itr']
moe_8_64_8lyr = data['64ept-8tpk-1itr-8lyr']
moe_8_64_12lyr = data['64ept-8tpk-1itr-12lyr']
coe_8_64 = data['64ept-8tpk-2itr']

# Define labels for left plot
left_columns = ['MoE(8/64) baseline', 'MoE(8/64) 8-layer', 'MoE(8/64) 12-layer', 'CoE-2(8/64)']
left_data = [moe_8_64, moe_8_64_8lyr, moe_8_64_12lyr, coe_8_64]

# Find the index where steps >= 100
start_idx = np.where(steps >= 100)[0]
start_idx = 0 if len(start_idx) == 0 else start_idx[0]

# Define colors and markers for left plot
left_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
left_markers = ['o', '^', 'D', 's']
left_linestyles = ['-', '-.', ':', '--']

# Create plot with log scales for selected columns
for i, (data_series, label) in enumerate(zip(left_data, left_columns)):
    ax1.loglog(steps[start_idx:], data_series[start_idx:], 
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
models = ['64ept-8tpk-1itr', '64ept-8tpk-2itr', '64ept-8tpk-1itr-8lyr', '64ept-8tpk-1itr-12lyr']
labels = ['MoE(8/64)\n(baseline)', 'CoE-2(8/64)\n4-layer', 'MoE(8/64)\n8-layer', 'MoE(8/64)\n12-layer']

# Parameters in MB
params = [
    544.51,          # MoE(8/64) baseline
    544.75,          # CoE-2(8/64)
    1089.01,         # MoE(8/64)-8layer
    1633.52,         # MoE(8/64)-12layer
]

# Memory in GB
memory = [
    11.70,           # MoE(8/64) baseline
    11.70,           # CoE-2(8/64)
    20.21,           # MoE(8/64)-8layer
    28.71,           # MoE(8/64)-12layer
]

# Training time in seconds
time = [
    989.50,          # MoE(8/64) baseline
    1755.13,         # CoE-2(8/64)
    1863.37,         # MoE(8/64)-8layer
    2734.07,         # MoE(8/64)-12layer
]

# Normalize the values relative to baseline MoE(8/64)
params_norm = [p/params[0]*100 for p in params]
memory_norm = [m/memory[0]*100 for m in memory]
time_norm = [t/time[0]*100 for t in time]

# Set width of bars
barWidth = 0.2

# Set positions of the bars on X axis
r1 = np.arange(len(models))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# 使用与左图相同的颜色
bar_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# Create grouped bars - 参数柱状图使用与左图相同的颜色
for i in range(len(models)):
    ax2.bar(r1[i], params_norm[i], width=barWidth, color=bar_colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.bar(r2[i], memory_norm[i], width=barWidth, color=bar_colors[i], alpha=0.5, edgecolor='black', linewidth=1)
    ax2.bar(r3[i], time_norm[i], width=barWidth, color=bar_colors[i], alpha=0.3, edgecolor='black', linewidth=1)

# 为图例添加自定义句柄
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='Parameters'),
    Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Memory'),
    Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Training Time')
]
ax2.legend(handles=legend_elements, frameon=True, fontsize=12, framealpha=0.7, edgecolor='#333333', loc='upper left')

# Show the reference line at 100%
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1)

# Add values on the bars
for i, (p, m, t) in enumerate(zip(params_norm, memory_norm, time_norm)):
    ax2.text(r1[i], p+1, f"{p:.1f}%", ha='center', va='bottom', fontsize=10)
    ax2.text(r2[i], m+1, f"{m:.1f}%", ha='center', va='bottom', fontsize=10)
    ax2.text(r3[i], t+1, f"{t:.1f}%", ha='center', va='bottom', fontsize=10)

# Highlight the reductions of CoE-2(8/48)-4layer using colors matching left plot
# ax2.annotate(f"-{100-params_norm[0]:.1f}% params", 
#             xy=(r1[0], params_norm[0]), 
#             xytext=(r1[0], params_norm[0]-15),
#             fontsize=11,
#             weight='bold',
#             color=bar_colors[0],
#             ha='center',
#             bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc=bar_colors[0], ec="none"))

# ax2.annotate(f"-{100-memory_norm[0]:.1f}% memory", 
#             xy=(r2[0], memory_norm[0]), 
#             xytext=(r2[0], memory_norm[0]-15),
#             fontsize=11,
#             weight='bold',
#             color=bar_colors[0],
#             ha='center',
#             bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc=bar_colors[0], ec="none"))

# ax2.annotate(f"-{100-time_norm[0]:.1f}% time", 
#             xy=(r3[0], time_norm[0]), 
#             xytext=(r3[0], time_norm[0]-15),
#             fontsize=11,
#             weight='bold',
#             color=bar_colors[0],
#             ha='center',
#             bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc=bar_colors[0], ec="none"))

# Add xticks for each model group with colors matching left plot
ax2.set_xticks([r + barWidth for r in range(len(models))])
# 设置x轴标签颜色与左图线条颜色一致
for i, label in enumerate(labels):
    ax2.text(i + barWidth, -30, label, color=bar_colors[i], ha='center', va='center', fontweight='bold', fontsize=12)
ax2.set_xticklabels(['' for _ in range(len(labels))])  # 清空原有标签

# Set labels and title for right plot
ax2.set_ylabel('Relative to MoE(8/64) baseline (%)', fontweight='bold', fontsize=14)
ax2.set_title('Resource Efficiency', fontweight='bold', fontsize=16)
ax2.set_ylim(0, 300)
ax2.set_facecolor('#f8f9fa')

# Add gridlines
ax2.grid(axis='y', linestyle='-', alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Main title for the entire figure
plt.suptitle('Compute Scaling: #Iteration (CoE) > #MoE Layers', 
            fontweight='bold', fontsize=18, y=0.98)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('plot_lyr.png', dpi=300, bbox_inches='tight')
plt.show()