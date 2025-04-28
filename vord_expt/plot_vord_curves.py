# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:25:32 2025

@author: User
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import interpolate
import matplotlib.patches as patches

root = "/home/dex/Downloads/"
def moving_average(data, smoothing_weight=0.99, start=0):
    """Calculates the exponentially weighted moving average."""
    smoothed = []
    last = data[start]  # Initialize with the first value
    for point in data:
        smoothed_val = (1 - smoothing_weight) * point + smoothing_weight * last
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# File paths
file_paths_vord = {
    'Base': root + "/run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion-tag-train_log_vord_loss_margin.csv",
    'VORD 1':root + "/run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion-correct-vit-tag-train_log_vord_loss_margin.csv",
    'VORD 2': root + "/run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-correct-vit-tag-train_log_vord_loss_margin.csv",
}

file_paths_gradnorm ={
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion-tag-train_grad_norm.csv",
    'VORD 1':root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion-tag-train_grad_norm.csv",
    'VORD 2': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-tag-train_grad_norm.csv",
}

file_paths_train_acc ={
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion-tag-train_token_acc.csv",
    'VORD 1':root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion-tag-train_token_acc.csv",
    'VORD 2': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-tag-train_token_acc.csv",
}

file_paths_xent = {
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion-tag-train_log_xent_loss.csv",
    'VORD 1': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion-tag-train_log_xent_loss.csv",
    'VORD 2': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-tag-train_log_xent_loss.csv",
}

file_paths_violations = {
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-false-low-tag-train_log_num_violations.csv",
    'VORD': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-true-low-tag-train_log_num_violations.csv",
}

file_paths_ordinal_ent = {
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-false-low-tag-train_log_ordinal_ent.csv",
    'VORD': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-true-low-tag-train_log_ordinal_ent.csv",
}

file_paths_SNR = {
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-false-tag-train_log_signal_noise_ratio.csv",
    'VORD': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-true-tag-train_log_signal_noise_ratio.csv",
}

margin_file_path = root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-vord-true-low-tag-train_log_margin.csv"
plt.rcParams.update({'font.size': 25})

def plot_loss(file_paths, loss_type, save_file, smoothing_weight=0.99, start=7):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(6,5))
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        
        if start is not None:
            plt.plot(df[start:-1]['Step'], moving_average(df[start: -1]['Value'], smoothing_weight, start=start), '-' ,label=label)
        else:
            plt.plot(df['Step'], moving_average(df['Value'], smoothing_weight), label=label)

    plt.xlabel('Steps')
    plt.ylabel(loss_type)
    #plt.title('Train Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig("./asset/" + save_file, format="pdf", bbox_inches="tight")
    if loss_type == "VORD Loss":
        plt.ylim(0.15, 0.55)
    plt.show()
    
def plot_margin(file_paths, margin_file_path, loss_type, save_file, smoothing_weight=0.99, start=10):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(6,5))
    margin_df = pd.read_csv(margin_file_path)
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        if start is not None:
            plt.plot(df[start:-1]['Step'], moving_average(df[start: -1]['Value']/margin_df[start: -1]['Value'], smoothing_weight, start=start), '-' ,label=label)
        else:
            plt.plot(df['Step'], moving_average(df['Value']/margin_df['Value'], smoothing_weight), label=label)

    plt.xlabel('Steps')
    plt.ylabel(loss_type)
    #plt.title('Train Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig("./asset/" + save_file, format="pdf", bbox_inches="tight")
    if loss_type == "VORD Loss":
        plt.ylim(0.15, 0.55)
    plt.show()

def draw_cluster(axs, x, y, c='cyan'):
    points = np.stack([x, y ]).T
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    #spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
    spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    
    axs.fill(interp_x, interp_y, alpha=0.1, color=c)

def plot_scatter_patterns(x_file_paths, y_file_paths, x_label, y_label, save_file, smoothing_weight=0.9, start=0):
    """
    Plots scatter patterns between two metrics over steps.
    """
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    
    colors = ['b', 'y']
    
    for i, key in enumerate(x_file_paths):
        x_file_path = x_file_paths[key]
        y_file_path = y_file_paths[key]

        x_df = pd.read_csv(x_file_path)
        y_df = pd.read_csv(y_file_path)
        
        # Apply smoothing
        smoothed_x = moving_average(x_df[start:-1]['Value'], smoothing_weight, start=start)
        smoothed_y = moving_average(y_df[start:-1]['Value'], smoothing_weight, start=start)

        # Note: The original scatter plot had `3000 - smoothed_x`.
        # Assuming this is a desired transformation, apply it here.
        # If not, use `smoothed_x` directly.
        plt.scatter(3000 - smoothed_x, smoothed_y, label=key, alpha=0.8)
        draw_cluster(ax, 3000 - smoothed_x, smoothed_y, c=colors[i])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    plt.savefig("./asset/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()


#plot_loss(file_paths_vord, 'VORD Loss', save_file="deepseek_vl_vord.pdf", smoothing_weight=0.5)
#plot_loss(file_paths_gradnorm, 'Gradient Norm', save_file="deepseek_vl_gradnorm.pdf", smoothing_weight=0.90)
#plot_loss(file_paths_train_acc, 'Training Accuracy', save_file="deepseek_vl_train_acc.pdf", smoothing_weight=0.95)
#plot_loss(file_paths_xent, 'X-ent Loss', save_file="deepseek_vl_xent.pdf", smoothing_weight=0.85)

plot_loss(file_paths_violations, '# of violations', save_file="paligemma_violations.pdf", smoothing_weight=0.97, start=0)
plot_loss(file_paths_ordinal_ent, 'KL Divergence', save_file="paligemma_ordinal_ent.pdf", smoothing_weight=0.98, start=25)
plot_loss(file_paths_SNR, 'Signal-to-Noise ratio', save_file="paligemma_snr.pdf", smoothing_weight=0.95, start=0)
plot_margin(file_paths_ordinal_ent, margin_file_path, 'KL Divergence/mθ', save_file="paligemma_ordinal_ent_margin.pdf", smoothing_weight=0.98, start=30)
plot_scatter_patterns(file_paths_violations, file_paths_SNR, "Violations reduced", "Signal-to-Noise ratio", save_file="paligemma_snr_violations.pdf", smoothing_weight=0.9, start=35)

#%%
# Data
methods = ['Baseline', 'VORD']
scores1 = np.array([1399.85, 1580.75]) # First score for each method
scores2 = np.array([225.00, 320.00])   # Second score for each method

# Calculate the combined and scaled score as specified
scaling_factor = 2800
scores = (scores1 + scores2) / scaling_factor
scores *= 100
print(f"Baseline Combined Scaled Score: {scores[0]:.4f}")
print(f"VORD Combined Scaled Score: {scores[1]:.4f}")

# Bar width (adjust as needed for clarity)
bar_width = 0.4 # Slightly smaller width might look better when not adjacent

# Positions of the bars on the x-axis
x_positions = [0.5, 1.0] # [0, 1]

# Create the plot
fig, ax = plt.subplots(figsize=(4,9))
ax = plt.gca() # Get the current axes for grid control

bar_baseline = plt.bar(x_positions[0], scores[0], color='skyblue', width=bar_width, edgecolor='grey', label='Base')
bar_vord = plt.bar(x_positions[1], scores[1], color='lightcoral', width=bar_width, edgecolor='black', linewidth=2.5, label='VORD')


for i, bar in enumerate([bar_baseline, bar_vord]):
    yval = bar[0].get_height()# Get height from the rectangle object
    plt.text(x_positions[i], yval, f'{yval:.1f}%', ha='center', va='bottom') # Using .3f for slightly more precision


# plt.xlabel('Method', fontweight='bold') # Removed as per your snippet
plt.ylabel('Accuracy (%)') # Label from your snippet

plt.xticks([])
plt.ylim(45, 70)
plt.yticks([50, 60])

ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

fig.subplots_adjust(bottom=0.175, wspace=0.3, hspace=0.0)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', ncol=1, fontsize=26)
plt.show()
#%%
plt.rcParams.update({'font.size': 15})

# Define the data
x_noise = [0, 250, 500, 750, 999]
scores = np.array([[1184.08, 242.5],
                   [1236.75, 269.64],
                   [1260.75, 320],
                   [1260.75, 301.1],
                   [1286.4, 281.78]])

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
axs[0].plot(x_noise, scores[:, 0], '-o', label='Column 1 Scores')
axs[1].plot(x_noise, scores[:, 1], '-o', label='Column 2 Scores')
axs[2].plot(x_noise, scores.sum(-1), '-o', label='Column 2 Scores')

axs[0].set_ylabel('Perception Score')
axs[1].set_ylabel('Reasoning Score')
axs[2].set_ylabel('Total Score')

for i in range(3):
    axs[i].grid('True')
    axs[i].set_xlabel('Corruption steps')

plt.tight_layout()
plt.show()