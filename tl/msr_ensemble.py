# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from utils.dataloader import data_process

def run_full_evaluation():
    # ================= 配置区域 =================
    datasets = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    method = 'T-TIME'
    
    # 定义要对比的两个文件名后缀
    # 对应 ttime.py 中的 experiment_configs
    suffix_baseline = "_D0.0_a0.0" 
    suffix_sr       = "_D0.5_a0.0"
    
    # 集成权重
    w1, w2 = 0.5, 0.5
    # ===========================================

    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    for data_name in datasets:
        print(f"\n{'='*15} Processing Dataset: {data_name} {'='*15}")
        
        # 1. 获取该数据集的参数和真实标签
        if data_name == 'BNCI2014001': trial_num, N = 144, 9
        elif data_name == 'BNCI2014002': trial_num, N = 100, 14
        elif data_name == 'BNCI2015001': trial_num, N = 200, 12
        elif data_name == 'BNCI2014001-4': trial_num, N = 288, 9
        
        # 加载标签
        _, y_all, _, _, _, _ = data_process(data_name)
        
        # 存储该数据集下每个被试的平均准确率
        # 结构: [subj_id, acc_baseline, acc_sr, acc_ensemble]
        subject_results = {
            'Baseline': np.zeros(N),
            'SR': np.zeros(N),
            'Ensemble': np.zeros(N)
        }
        
        valid_seeds = 0

        # 2. 遍历种子，累加预测结果
        for seed in seeds:
            path_base = f'./logs/{data_name}_{method}_seed_{seed}{suffix_baseline}_pred.csv'
            path_sr   = f'./logs/{data_name}_{method}_seed_{seed}{suffix_sr}_pred.csv'
            
            if not os.path.exists(path_base) or not os.path.exists(path_sr):
                # print(f"Skipping Seed {seed} (File missing)")
                continue
            
            valid_seeds += 1
            
            # 读取概率
            preds_base = pd.read_csv(path_base, header=None).to_numpy() # (N, trial_num)
            preds_sr   = pd.read_csv(path_sr, header=None).to_numpy()   # (N, trial_num)
            
            # 计算集成概率
            preds_ens = w1 * preds_base + w2 * preds_sr
            
            # 对每个被试计算准确率并累加
            for subj in range(N):
                start_idx = trial_num * subj
                y_true = y_all[start_idx : start_idx + trial_num]
                
                # 计算三个方法的准确率
                acc_b = accuracy_score(y_true, np.where(preds_base[subj]>=0.5, 1, 0))
                acc_s = accuracy_score(y_true, np.where(preds_sr[subj]>=0.5, 1, 0))
                acc_e = accuracy_score(y_true, np.where(preds_ens[subj]>=0.5, 1, 0))
                
                subject_results['Baseline'][subj] += acc_b
                subject_results['SR'][subj] += acc_s
                subject_results['Ensemble'][subj] += acc_e

        if valid_seeds == 0:
            print("No valid data found for this dataset.")
            continue

        # 3. 取平均 (除以种子数) 并转换为百分比
        for key in subject_results:
            subject_results[key] = (subject_results[key] / valid_seeds) * 100
            
        # 打印摘要
        print(f"Processed {valid_seeds} seeds.")
        avg_b = np.mean(subject_results['Baseline'])
        avg_s = np.mean(subject_results['SR'])
        avg_e = np.mean(subject_results['Ensemble'])
        print(f"Avg Acc -> Baseline: {avg_b:.2f}% | SR: {avg_s:.2f}% | Ensemble: {avg_e:.2f}%")

        # 4. 生成折线对比图
        plt.figure(figsize=(12, 6))
        subjects = np.arange(N)
        
        # 绘制三条线
        plt.plot(subjects, subject_results['Baseline'], 'o--', label='Baseline (No SR)', color='gray', alpha=0.7)
        plt.plot(subjects, subject_results['SR'], 's-', label='With SR (D=0.5)', color='#1f77b4') # Blue
        plt.plot(subjects, subject_results['Ensemble'], '^-', label='Ensemble', color='#d62728', linewidth=2.5) # Red
        
        # 美化图表
        plt.title(f"Performance Comparison - {data_name} (Avg of {valid_seeds} seeds)", fontsize=14)
        plt.xlabel("Subject ID", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xticks(subjects, [f'S{i}' for i in subjects])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 保存图片
        save_path = f'./logs/Comparison_Chart_{data_name}.png'
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to: {save_path}")
        plt.close()

if __name__ == '__main__':
    run_full_evaluation()