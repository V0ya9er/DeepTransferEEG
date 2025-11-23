# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : ttime.py
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.benchmark = True
import pandas as pd
import csv
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb, cal_score_online
from utils.alg_utils import *
from utils.sr_utils import apply_bistable_sr_torch
from scipy.linalg import fractional_matrix_power
from utils.loss import Entropy
from sklearn.metrics import roc_auc_score, accuracy_score

import gc
import sys
import time


def TTIME(loader, model, args, balanced=True):
    # "T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs"
    # IEEE Transactions on Biomedical Engineering
    # Note that the ensemble experiment is separately implemented in ttime_ensemble.py, using recorded test prediction.

    if balanced == False and args.data_name == 'BNCI2014001-4':
        print('ERROR, imbalanced multi-class not implemented')
        sys.exit(0)

    y_true = []
    y_pred = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize test reference matrix for Incremental EA
    if args.align:
        R_tensor = None # 我们将 R 初始化为 None

    if not balanced:
        zk_arrs = np.zeros(2)
        c = 4

    iter_test = iter(loader)
    
    # 预计算 MSR 时间步长
    msr_dt = 1.0 / args.sample_rate
    # 确定我们的目标设备 (从模型参数中获取, e.g., 'cuda:0')
    device = next(model.parameters()).device
    
    # 在循环开始前预分配内存
    # 假设 inputs_shape 是 (1, C, T)
    total_samples = len(loader)
    # 预分配一个全零的大张量在 GPU 上
    data_cum = torch.zeros((total_samples, 1, args.chn, args.time_sample_num), 
                           device=device, dtype=torch.float32)
    
    # 维护一个计数器
    current_idx = 0
    
    # loop through test data stream one by one
    for i in range(len(loader)):
        #################### Phase 1: target label prediction ####################
        model.eval()
        data = next(iter_test)
        inputs_raw = data[0]
        labels = data[1]
        
        # ==========================================================
        # ================ 在 TTIME 循环中应用 SR ================
        # (在 EA 和模型预测之前)
        # ==========================================================
# ==========================================================
        # ================ 修正：调用 PyTorch MSR ===================
        # ==========================================================
        
        # 1. 挤压(Squeeze)掉多余的维度：(1, 1, 22, 1001) -> (1, 22, 1001)
        inputs_3d = inputs_raw.squeeze(1) # inputs_3d.shape 是 (1, 22, 1001)

        # 2. !! 将数据推送到 GPU !!
        inputs_3d_gpu = inputs_3d.to(device)
        
        # === 新增：信号衰减系数 (Input Scaling) ===
        # 将信号幅度缩小到原本的 1/10，使其变成“弱信号”
        # 这样势垒 (0.25) 对它来说就很高了，必须要噪声帮忙
        scaling_factor = 1.0  
        inputs_scaled_gpu = inputs_3d_gpu * scaling_factor
        
        if args.msr_noise_intensity > 0:
            # 生成与信号形状相同的噪声
            # 噪声强度直接由 msr_noise_intensity 控制
            # 这里不需要 dt，直接控制噪声的幅度标准差
            noise = torch.randn_like(inputs_scaled_gpu) * args.msr_noise_intensity
            
            inputs_sr_gpu = inputs_scaled_gpu + noise
        else:
            # D=0 时，直接原样输出
            inputs_sr_gpu = inputs_scaled_gpu
            
        #==========================================================
        # ======================= SR 结束 ==========================
        # ==========================================================
        # if i == 0: # 为了不刷屏，我们只画第一个试次(Trial)的图
        #     import matplotlib.pyplot as plt
        #     import os
            
        #     # 获取第一个通道的数据 (转回 CPU 以便绘图)
        #     # 对比：缩放后的原始信号 vs MSR处理后的信号
        #     orig_wave = inputs_scaled_gpu[0, 0, :].detach().cpu().numpy()
        #     sr_wave = inputs_sr_gpu[0, 0, :].detach().cpu().numpy()
            
        #     plt.figure(figsize=(12, 8))
            
        #     # 子图1: 原始信号 (缩放后)
        #     plt.subplot(2, 1, 1)
        #     plt.plot(orig_wave, color='blue', alpha=0.7)
        #     plt.title(f"Original Input (Scaled by {scaling_factor})")
        #     plt.grid(True, alpha=0.3)
            
        #     # 子图2: MSR 输出信号
        #     plt.subplot(2, 1, 2)
        #     plt.plot(sr_wave, color='red', alpha=0.7)
        #     plt.title(f"MSR Output (D={args.msr_noise_intensity}, a={args.msr_a}, b={args.msr_b})")
        #     plt.grid(True, alpha=0.3)
            
        #     plt.tight_layout()
            
        #     # 保存图片到 logs 文件夹
        #     save_path = f'./logs/signal_check_D{args.msr_noise_intensity}_a{args.msr_a}.png'
        #     plt.savefig(save_path)
        #     print(f"\n[Diagnostic] Signal plot saved to: {save_path}")
        #     print("[Diagnostic] Please check this image to see if the signal is destroyed!")
            
            # (可选) 如果您想看完图就停止程序，可以取消下面这行的注释
            # sys.exit(0) 
        # =======================================================
        inputs = inputs_sr_gpu.unsqueeze(1)

        # 5. 在 GPU 上累积
        # 直接填入对应位置，不进行 cat 操作
        data_cum[i] = inputs # inputs 已经在 GPU 上


        # Incremental EA
        if args.align:
            start_time = time.time()
            
            # 1. 从 GPU 上的 data_cum 获取数据
            sample_test_gpu = data_cum[i].reshape(args.chn, args.time_sample_num)
            
            # 2. EA 计算 (GPU)
            R_tensor = EA_online_torch(sample_test_gpu, R_tensor, i)
            sqrtRefEA_torch = matrix_power_torch(R_tensor, -0.5)
            
            # 3. 矩阵乘法 (GPU)
            # 结果保存在 sample_test_aligned_gpu 中
            sample_test_aligned_gpu = torch.matmul(sqrtRefEA_torch, sample_test_gpu)

            EA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', pre-inference IEA finished time in ms:', np.round((EA_time - start_time) * 1000, 3))
            
            # 4. 恢复形状 (这里需要使用 sample_test_aligned_gpu)
            sample_test = sample_test_aligned_gpu.reshape(1, 1, args.chn, args.time_sample_num)
            
        else:
            sample_test = inputs

        # if args.data_env != 'local':
        #     sample_test = torch.from_numpy(sample_test).to(torch.float32).cuda()
        # else:
        #     sample_test = torch.from_numpy(sample_test).to(torch.float32)

        # !! sample_test 和 model 都在 GPU 上
        _, outputs = model(sample_test.to(torch.float32))

        softmax_out = nn.Softmax(dim=1)(outputs)

        # (!! Phase 1 的评估部分仍然需要 .cpu() 来存储结果 !!)
        outputs_cpu = outputs.float().cpu()
        labels_cpu = labels.float().cpu()
        _, predict = torch.max(outputs_cpu, 1)
        y_pred.append(softmax_out.detach().cpu().numpy())
        y_true.append(labels_cpu.item())

        #################### Phase 2: target model update ####################
        model.train()
        # sliding batch
        if (i + 1) >= args.test_batch and (i + 1) % args.stride == 0:
            if args.align:
                # !! TTA 批处理 (!! 现在完全在 GPU 上 !!)
                
                # 1. 从 GPU 上的 data_cum 获取批次
                batch_gpu = data_cum[i - args.test_batch + 1:i + 1] # (B, 1, C, T)
                
                # 2. 准备批次进行矩阵乘法
                batch_squeezed = batch_gpu.squeeze(1) # (B, C, T)
                
                # 3. 准备对齐矩阵 (C, C) -> (B, C, C)
                sqrtRefEA_batch = sqrtRefEA_torch.unsqueeze(0).expand(batch_squeezed.shape[0], -1, -1) 
                
                # 4. 在 GPU 上对齐批次
                aligned_batch = torch.matmul(sqrtRefEA_batch, batch_squeezed) # (B, C, T)
                
                # 5. 恢复形状 (仍在 GPU 上)
                batch_test = aligned_batch.unsqueeze(1)
            else:
                # (如果不用 EA, data_cum 已经在 GPU 上)
                batch_test = data_cum[i - args.test_batch + 1:i + 1]
                # batch_test = batch_test.reshape(args.test_batch, 1, batch_test.shape[2], batch_test.shape[3])

            # if args.data_env != 'local':
            #     batch_test = torch.from_numpy(batch_test).to(torch.float32).cuda()
            # else:
            #     batch_test = torch.from_numpy(batch_test).to(torch.float32)

            start_time = time.time()
            for step in range(args.steps):

                # (!! 整个 TTA 更新循环现在都在 GPU 上 !!)
                _, outputs = model(batch_test.to(torch.float32)) # GPU
                outputs = outputs.float()

                args.epsilon = 1e-5
                softmax_out = nn.Softmax(dim=1)(outputs / args.t)
                # Conditional Entropy Minimization loss
                CEM_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)

                if balanced:
                    # Marginal Distribution Regularization loss
                    MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                    loss = CEM_loss + MDR_loss
                else:
                    # Adaptive Marginal Distribution Regularization
                    qk = torch.zeros((args.class_num, )).to(torch.float32)
                    for k in range(args.class_num):
                        qk[k] = msoftmax[k] / (c + zk_arrs[k])
                    sum_qk = torch.sum(qk)
                    normed_qk = qk / sum_qk
                    AMDR_loss = torch.sum(normed_qk * torch.log(normed_qk + args.epsilon))
                    loss = CEM_loss + AMDR_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            TTA_time = time.time()
            if args.calc_time:
                print('sample ', str(i), ', post-inference model update finished in ms:', np.round((TTA_time - start_time) * 1000, 3))

            if not balanced:
                if i + 1 == args.test_batch:
                    args.pred_thresh = 0.7
                    pl = torch.max(softmax_out, 1)[1]
                    for l in range(args.test_batch):
                        if pl[l] == 0:
                            if softmax_out[l][0] > args.pred_thresh:
                                zk_arrs[0] += 1
                        elif pl[l] == 1:
                            if softmax_out[l][1] > args.pred_thresh:
                                zk_arrs[1] += 1
                        else:
                            print('ERROR in pseudo labeling!')
                            sys.exit(0)
                else:
                    # update confident prediction ids for current test sample
                    pl = torch.max(softmax_out, 1)[1]
                    if pl[-1] == 0:
                        if softmax_out[-1][0] > args.pred_thresh:
                            zk_arrs[0] += 1
                    elif pl[-1] == 1:
                        if softmax_out[-1][1] > args.pred_thresh:
                            zk_arrs[1] += 1
                    else:
                        print('ERROR in pseudo labeling!')

        model.eval()

    if balanced:
        _, predict = torch.max(torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num), 1)
        pred = torch.squeeze(predict).float()
        score = accuracy_score(y_true, pred)
        if args.data_name == 'BNCI2014001-4':
            y_pred = np.array(y_pred).reshape(-1, )  # multiclass
        else:
            y_pred = np.array(y_pred).reshape(-1, args.class_num)[:, 1]  # binary
    else:
        predict = torch.from_numpy(np.array(y_pred)).to(torch.float32).reshape(-1, args.class_num)
        y_pred = np.array(predict).reshape(-1, args.class_num)[:, 1]  # binary
        score = roc_auc_score(y_true, y_pred)

    return score * 100, y_pred


def train_target(args):
    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    if args.max_epoch == 0:
        if args.align:
            if args.data_env != 'local':
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt'))
            else:
                base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.backbone) +
                    '_S' + str(args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            features_source, outputs_source = base_network(inputs_source)

            classifier_loss = criterion(outputs_source, labels_source)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                if args.balanced:
                    acc_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                else:
                    acc_t_te, _ = cal_auc_comb(dset_loaders["Target-Imbalanced"], base_network, args=args)
                    log_str = 'Task: {}, Iter:{}/{}; Offline-EA AUC = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

        print('saving model...')
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(
                       args.idt) + '_seed' + str(args.SEED) + extra_string + '.ckpt')


    base_network.eval()

    score = cal_score_online(dset_loaders["Target-Online"], base_network, args=args)
    if args.balanced:
        log_str = 'Task: {}, Online IEA Acc = {:.2f}%'.format(args.task_str, score)
    else:
        log_str = 'Task: {}, Online IEA AUC = {:.2f}%'.format(args.task_str, score)
    args.log.record(log_str)
    print(log_str)

    print('executing TTA...')

    if args.balanced:
        acc_t_te, y_pred = TTIME(dset_loaders["Target-Online"], base_network, args=args, balanced=True)
        log_str = 'Task: {}, TTA Acc = {:.2f}%'.format(args.task_str, acc_t_te)
    else:
        acc_t_te, y_pred = TTIME(dset_loaders["Target-Online-Imbalanced"], base_network, args=args, balanced=False)
        log_str = 'Task: {}, TTA AUC = {:.2f}%'.format(args.task_str, acc_t_te)
    args.log.record(log_str)
    print(log_str)

    if args.balanced:
        print('Test Acc = {:.2f}%'.format(acc_t_te))

    else:
        print('Test AUC = {:.2f}%'.format(acc_t_te))

    torch.save(base_network.state_dict(), './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
        args.SEED) + extra_string + '_adapted' + '.ckpt')

    # save the predictions for ensemble
    # 修正：将 MSR 参数加入文件名，防止覆盖
    file_suffix = f"_D{args.msr_noise_intensity}_a{args.msr_a}_pred.csv"
    save_path = './logs/' + str(args.data_name) + '_' + str(args.method) + '_seed_' + str(args.SEED) + file_suffix
    
    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred)

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te





if __name__ == '__main__':
    import time
    
    # ================= 实验配置区域 =================
    # 1. 定义要跑的数据集
    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001', 'BNCI2014001-4']
    
    # 2. 定义对比实验组 (根据您之前的最佳实践)
    # Group 1: 基准 (No SR)
    # Group 2: 增强 (Additive Noise, D=0.5)
    experiment_configs = [
        {'D': 0.0, 'a': 0.0, 'b': 0.0, 'desc': 'Baseline'}, 
        {'D': 0.5, 'a': 0.0, 'b': 0.0, 'desc': 'SR_Noise'},
    ]
    
    # 3. 运行所有种子
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # ==============================================

    total_start_time = time.time()

    for config in experiment_configs:
        cur_D, cur_a, cur_b = config['D'], config['a'], config['b']
        print(f"\n\n{'#'*40}")
        print(f"### Starting Experiment Group: {config['desc']} (D={cur_D})")
        print(f"{'#'*40}")

        for data_name in data_name_list:
            # === 数据集参数配置 (保持原逻辑) ===
            if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
            if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
            if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
            if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248

            # 固定参数
            args = argparse.Namespace(feature_deep_dim=feature_deep_dim, align=True, lr=0.001, t=2, max_epoch=0,
                                      trial_num=trial_num, time_sample_num=time_sample_num, sample_rate=sample_rate,
                                      N=N, chn=chn, class_num=class_num, stride=1, steps=1, calc_time=False,
                                      paradigm=paradigm, test_batch=8, data_name=data_name, balanced=True,
                                      data=data_name, # 修复属性缺失
                                      
                                      # 动态 MSR 参数
                                      msr_noise_intensity=cur_D, msr_a=cur_a, msr_b=cur_b)

            args.method = 'T-TIME'
            args.backbone = 'EEGNet'
            args.batch_size = 32

            try:
                device_id = str(sys.argv[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id
                args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
            except:
                args.data_env = 'local'
            
            print(f"\n>>> Dataset: {data_name} | Config: D={cur_D}")

            for s in seed_list: 
                args.SEED = s
                fix_random_seed(args.SEED)
                torch.backends.cudnn.deterministic = True
                
                # 日志设置
                args.local_dir = './data/' + str(data_name) + '/'
                args.result_dir = './logs/'
                # 这里的日志文件主要用于调试，不需要太关注，重点是生成的 csv
                my_log = LogRecord(args) 
                my_log.log_init() # 可以注释掉以减少垃圾文件，或者保留

                sub_acc_all = np.zeros(N)
                for idt in range(N):
                    args.idt = idt
                    source_str = 'Except_S' + str(idt)
                    target_str = 'S' + str(idt)
                    args.task_str = source_str + '_2_' + target_str
                    
                    args.log = my_log # 修复属性缺失
                    
                    # print(f"  Run: {data_name} | Seed {s} | Sub {idt} ...", end='\r')
                    sub_acc_all[idt] = train_target(args)
                
                # print(f"  Run: {data_name} | Seed {s} | Done. Avg: {np.mean(sub_acc_all):.2f}%")

    print(f"\nAll experiments finished in {(time.time() - total_start_time)/60:.1f} minutes.")
    print("Now please run: python ./tl/msr_ensemble.py")