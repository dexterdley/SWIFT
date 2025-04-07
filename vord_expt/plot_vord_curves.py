# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:25:32 2025

@author: User
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

root = "/home/dex/Downloads/"
def moving_average(data, smoothing_weight=0.99):
    """Calculates the exponentially weighted moving average."""
    smoothed = []
    last = data[0]  # Initialize with the first value
    for point in data:
        smoothed_val = (1 - smoothing_weight) * point + smoothing_weight * last
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# File paths
file_paths_vord = {
    #'Base': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord0-max-mix-tag-train_log_vord_loss.csv",
    #'VORD 1': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-tag-train_log_vord_loss.csv",
    #'VORD 2': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-tag-train_log_vord_loss.csv",
    #'VORD 1 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-margin-tag-train_log_vord_loss.csv",
    #'VORD 2 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-margin-tag-train_log_vord_loss.csv",
    #'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-newvord0-margin-diffusion-tag-train_log_vord_loss_margin.csv",
    #'VORD 1': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-newvord1-grad-margin-diffusion-tag-train_log_vord_loss_margin.csv",
    #'VORD 2': root + "run-AI-ModelScope_LLaVA-Instruct-150K_paligemma-3b-pt-224-finetune-newvord2-grad-margin-diffusion-tag-train_log_vord_loss_margin.csv",
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
    #'Base': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord0-max-mix-tag-train_log_xent_loss.csv",
    #'VORD 1': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-tag-train_log_xent_loss.csv",
    #'VORD 2': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-tag-train_log_xent_loss.csv",
    #'VORD 1 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-margin-tag-train_log_xent_loss.csv",
    #'VORD 2 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-margin-tag-train_log_xent_loss.csv",
    'Base': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion-tag-train_log_xent_loss.csv",
    'VORD 1': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion-tag-train_log_xent_loss.csv",
    'VORD 2': root + "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-tag-train_log_xent_loss.csv",
}

plt.rcParams.update({'font.size': 25})

def plot_loss(file_paths, loss_type, save_file, smoothing_weight=0.99):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(6,5))
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
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

plot_loss(file_paths_vord, 'VORD Loss', save_file="deepseek_vl_vord.pdf", smoothing_weight=0.5)
plot_loss(file_paths_gradnorm, 'Gradient Norm', save_file="deepseek_vl_gradnorm.pdf", smoothing_weight=0.90)
plot_loss(file_paths_train_acc, 'Training Accuracy', save_file="deepseek_vl_train_acc.pdf", smoothing_weight=0.95)
plot_loss(file_paths_xent, 'X-ent Loss', save_file="deepseek_vl_xent.pdf", smoothing_weight=0.85)
#%%
'''
import seaborn as sns

# Increase font size for all plots
plt.rcParams.update({'font.size': 16})

vord1_grad_norm = "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-full-finetune-vord1-margin-diffuse-tag-train_grad_norm.csv"
vord2_grad_norm = "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-full-finetune-vord2-margin-diffuse-tag-train_grad_norm.csv"
vord1_loss = "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-full-finetune-vord1-margin-diffuse-tag-train_log_vord_loss.csv"
vord2_loss = "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-full-finetune-vord2-margin-diffuse-tag-train_log_vord_loss.csv"

def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

vord1_grad_norm_value = pd.read_csv(vord1_grad_norm)['Value']
vord2_grad_norm_value = pd.read_csv(vord2_grad_norm)['Value']
vord1_loss_value = pd.read_csv(vord1_loss)['Value']
vord2_loss_value = pd.read_csv(vord2_loss)['Value']

# Normalize the values
vord1_grad_norm_norm = normalize_series(vord1_grad_norm_value)
vord2_grad_norm_norm = normalize_series(vord2_grad_norm_value)
vord1_loss_norm = normalize_series(vord1_loss_value)
vord2_loss_norm = normalize_series(vord2_loss_value)

# Calculate the correlation for vord1 (using original values for correlation)
correlation_vord1 = vord1_loss_value.corr(vord1_grad_norm_value)
print(f"Correlation between loss and gradient norm for vord1: {correlation_vord1:.4f}")

# Calculate the correlation for vord2 (using original values for correlation)
correlation_vord2 = vord2_loss_value.corr(vord2_grad_norm_value)
print(f"Correlation between loss and gradient norm for vord2: {correlation_vord2:.4f}")

# --- Combined Plotting ---
plt.figure(figsize=(16, 10))

# Plot for vord1 (Normalized Scatter)
plt.subplot(2, 2, 1)
sns.scatterplot(x=vord1_grad_norm_norm, y=vord1_loss_norm)
plt.title(f"Correlation (vord1): {correlation_vord1:.4f} (Normalized)")
plt.xlabel("Normalized Gradient Norm")
plt.ylabel("Normalized Loss")
plt.grid(True)

# Plot for vord2 (Normalized Scatter)
plt.subplot(2, 2, 2)
sns.scatterplot(x=vord2_grad_norm_norm, y=vord2_loss_norm)
plt.title(f"Correlation (vord2): {correlation_vord2:.4f} (Normalized)")
plt.xlabel("Normalized Gradient Norm")
plt.ylabel("Normalized Loss")
plt.grid(True)

# Plot for vord1 (Line Plots)
plt.subplot(2, 2, 3)
plt.plot(vord1_grad_norm_value.index, vord1_grad_norm_norm, label='Gradient Norm', color='blue')
plt.plot(vord1_loss_value.index, vord1_loss_norm, label='Loss', color='red')
plt.xlabel("Steps")
plt.ylabel("Value")
plt.title(f"vord1 - Loss and Gradient Norm")
plt.grid(True)
plt.legend()

# Plot for vord2 (Line Plots)
plt.subplot(2, 2, 4)
plt.plot(vord2_grad_norm_value.index, vord2_grad_norm_norm, label='Gradient Norm', color='blue')
plt.plot(vord2_loss_value.index, vord2_loss_norm, label='Loss', color='red')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"vord2 - Loss and Gradient Norm")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
'''