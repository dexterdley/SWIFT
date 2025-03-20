# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:25:32 2025

@author: User
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    'Base': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord0-max-mix-tag-train_log_vord_loss.csv",
    'VORD 1': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-tag-train_log_vord_loss.csv",
    'VORD 2': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-tag-train_log_vord_loss.csv",
    'VORD 1 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-margin-tag-train_log_vord_loss.csv",
    'VORD 2 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-margin-tag-train_log_vord_loss.csv",
}

file_paths_xent = {
    'Base': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord0-max-mix-tag-train_log_xent_loss.csv",
    'VORD 1': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-tag-train_log_xent_loss.csv",
    'VORD 2': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-tag-train_log_xent_loss.csv",
    'VORD 1 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord1-max-mix-margin-tag-train_log_xent_loss.csv",
    'VORD 2 + m': "run-AI-ModelScope_LLaVA-Instruct-150K_deepseek-vl-7b-finetune-vord2-max-mix-margin-tag-train_log_xent_loss.csv",
}

def plot_loss(file_paths, loss_type, smoothing_weight=0.99):
    """Plots the loss over steps for given file paths."""
    plt.figure()
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        plt.plot(df['Step'], moving_average(df['Value'], smoothing_weight), label=label)

    plt.xlabel('Step')
    plt.ylabel(f'{loss_type} Loss')
    plt.title(f'Train Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(file_paths_vord, 'VORD')
plot_loss(file_paths_xent, 'X-ent', smoothing_weight=0.8)