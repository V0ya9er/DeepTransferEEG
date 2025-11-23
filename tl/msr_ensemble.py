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
    
    # 定义要对比的两个文件名后缀 (对应 ttime.py 中的配置)
    suffix_baseline = "_D0.0_a0.0" 
    suffix_sr       = "_D0.5_a0.0"
    
    # 集成权重
    w1, w2 = 0.5, 0.5
    # ===========================================

    # 设置更美观的绘图风格
    sns.set_theme(style="whitegrid", palette="muted")

    for data_name in datasets:
        print(f"\n{'='*15} Processing Dataset: {data_name} {'='*15}")
        
        # 1. 获取该数据集的参数
        if data_name == 'BNCI2014001': trial_num, N, n_classes = 144, 9, 2
        elif data_name == 'BNCI2014002': trial_num, N, n_classes = 100, 14, 2
        elif data_name == 'BNCI2015001': trial_num, N, n_classes = 200, 12, 2
        elif data_name == 'BNCI2014001-4': trial_num, N, n_classes = 288, 9, 4
        
        # 加载标签
        try:
            _, y_all, _, _, _, _ = data_process(data_name)
        except Exception as e:
            print(f"Error loading data for {data_name}: {e}")
            continue
        
        # 存储累积结果
        subject_results = {
            'Baseline': np.zeros(N),
            'SR': np.zeros(N),
            'Ensemble': np.zeros(N)
        }
        
        valid_seeds = 0

        # 2. 遍历种子
        for seed in seeds:
            path_base = f'./logs/{data_name}_{method}_seed_{seed}{suffix_baseline}_pred.csv'
            path_sr   = f'./logs/{data_name}_{method}_seed_{seed}{suffix_sr}_pred.csv'
            
            if not os.path.exists(path_base) or not os.path.exists(path_sr):
                continue
            
            valid_seeds += 1
            
            # 读取数据
            preds_base_raw = pd.read_csv(path_base, header=None).to_numpy()
            preds_sr_raw   = pd.read_csv(path_sr, header=None).to_numpy()
            
            # 对每个被试计算准确率
            for subj in range(N):
                start_idx = trial_num * subj
                y_true = y_all[start_idx : start_idx + trial_num]
                
                p_base = preds_base_raw[subj]
                p_sr = preds_sr_raw[subj]
                
                # 处理多分类 vs 二分类
                if n_classes == 2:
                    p_ens = w1 * p_base + w2 * p_sr
                    pred_b = np.where(p_base >= 0.5, 1, 0)
                    pred_s = np.where(p_sr >= 0.5, 1, 0)
                    pred_e = np.where(p_ens >= 0.5, 1, 0)
                else:
                    p_base = p_base.reshape(-1, n_classes)
                    p_sr = p_sr.reshape(-1, n_classes)
                    p_ens = w1 * p_base + w2 * p_sr
                    pred_b = np.argmax(p_base, axis=1)
                    pred_s = np.argmax(p_sr, axis=1)
                    pred_e = np.argmax(p_ens, axis=1)
                
                subject_results['Baseline'][subj] += accuracy_score(y_true, pred_b)
                subject_results['SR'][subj] += accuracy_score(y_true, pred_s)
                subject_results['Ensemble'][subj] += accuracy_score(y_true, pred_e)

        if valid_seeds == 0:
            print("No valid data found.")
            continue

        # 3. 取平均并转百分比
        for key in subject_results:
            subject_results[key] = (subject_results[key] / valid_seeds) * 100
            
        print(f"Processed {valid_seeds} seeds.")
        
        # 准备绘图数据
        subjects_labels = [f'S{i}' for i in range(N)]
        
        # ==============================================
        # 图表 1: 增量柱状图 (Improvement Delta) - 强烈推荐
        # ==============================================
        delta_sr = subject_results['SR'] - subject_results['Baseline']
        delta_ens = subject_results['Ensemble'] - subject_results['Baseline']
        
        df_delta = pd.DataFrame({
            'Subject': subjects_labels,
            'SR (D=0.5)': delta_sr,
            'Ensemble': delta_ens
        })
        # 融合数据以便绘图
        df_delta_melt = df_delta.melt('Subject', var_name='Method', value_name='Accuracy Change (%)')
        
        plt.figure(figsize=(12, 6))
        # 画基准线 0
        plt.axhline(0, color='black', linewidth=1, linestyle='-')
        
        # 绘制柱状图
        sns.barplot(data=df_delta_melt, x='Subject', y='Accuracy Change (%)', hue='Method', 
                    palette={'SR (D=0.5)': '#1f77b4', 'Ensemble': '#d62728'})
        
        plt.title(f"Improvement vs Baseline (D=0) - {data_name}", fontsize=14)
        plt.ylim(min(df_delta_melt['Accuracy Change (%)'].min(), -2) - 1, 
                 max(df_delta_melt['Accuracy Change (%)'].max(), 2) + 1)
        
        # 保存图片
        save_path_delta = f'./logs/Chart_Delta_{data_name}.png'
        plt.savefig(save_path_delta, dpi=300)
        print(f"Delta Chart saved to: {save_path_delta}")
        plt.close()

        # ==============================================
        # 图表 2: 分组柱状图 (Absolute Accuracy)
        # ==============================================
        df_abs = pd.DataFrame({
            'Subject': subjects_labels,
            'Baseline': subject_results['Baseline'],
            'SR (D=0.5)': subject_results['SR'],
            'Ensemble': subject_results['Ensemble']
        })
        df_abs_melt = df_abs.melt('Subject', var_name='Method', value_name='Accuracy (%)')
        
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_abs_melt, x='Subject', y='Accuracy (%)', hue='Method',
                    palette=['gray', '#1f77b4', '#d62728'])
        
        plt.title(f"Absolute Performance - {data_name}", fontsize=14)
        # 设置 Y 轴范围，让差异更明显 (从最低分-5 开始)
        min_acc = df_abs_melt['Accuracy (%)'].min()
        plt.ylim(max(0, min_acc - 10), 100)
        
        save_path_abs = f'./logs/Chart_Absolute_{data_name}.png'
        plt.savefig(save_path_abs, dpi=300)
        print(f"Absolute Chart saved to: {save_path_abs}")
        plt.close()

if __name__ == '__main__':
    run_full_evaluation()