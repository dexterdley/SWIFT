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
    margin_df = df = pd.read_csv(margin_file_path)
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

#plot_loss(file_paths_vord, 'VORD Loss', save_file="deepseek_vl_vord.pdf", smoothing_weight=0.5)
#plot_loss(file_paths_gradnorm, 'Gradient Norm', save_file="deepseek_vl_gradnorm.pdf", smoothing_weight=0.90)
#plot_loss(file_paths_train_acc, 'Training Accuracy', save_file="deepseek_vl_train_acc.pdf", smoothing_weight=0.95)
#plot_loss(file_paths_xent, 'X-ent Loss', save_file="deepseek_vl_xent.pdf", smoothing_weight=0.85)

plot_loss(file_paths_violations, '# of violations', save_file="paligemma_violations.pdf", smoothing_weight=0.97, start=0)
plot_loss(file_paths_ordinal_ent, 'KL Divergence', save_file="paligemma_ordinal_ent.pdf", smoothing_weight=0.98, start=25)
plot_margin(file_paths_ordinal_ent, margin_file_path, 'KL Divergence/mθ', save_file="paligemma_ordinal_ent_margin.pdf", smoothing_weight=0.98, start=30)
