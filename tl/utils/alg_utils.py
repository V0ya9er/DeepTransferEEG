# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : alg_utils.py
# Euclidean Alignment
# Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power


# numpy implementation, if error try EA_SPDsafe function
def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


# arithmetic mean only, SPD-safe
def EA_SPDsafe(x, epsilon=1e-6):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    n = len(x)
    C = np.zeros((x[0].shape[0], x[0].shape[0]))
    for X in x:
        C += X @ X.T
    R_bar = C / n
    trace = np.trace(R_bar)
    R_bar += epsilon * (trace / R_bar.shape[0]) * np.eye(R_bar.shape[0])

    eigvals, eigvecs = np.linalg.eigh(R_bar)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    ref = eigvecs @ D_inv_sqrt @ eigvecs.T

    XEA = ref @ x

    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)
    sample_num: int
        previous number of samples used to calculate R

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA
  
  
  # ==========================================================
# ============ 1. 添加以下三个新的 PyTorch 函数 =============
# ==========================================================

@torch.no_grad()
def matrix_power_torch(matrix, power, epsilon=1e-8):
    """
    计算对称矩阵的实数次幂 (PyTorch GPU 版本)。
    A = V @ Lambda @ V.T
    A^p = V @ Lambda^p @ V.T
    """
    # 使用 eigh 计算对称矩阵（协方差矩阵）的特征值和特征向量
    # R = V @ Lambda @ V.T
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # 添加 epsilon 以保证数值稳定性 (避免 0 或负数的 sqrt)
    eigenvalues_stable = eigenvalues.clamp(min=epsilon)
    
    # 计算 Lambda^p
    eigenvalues_powered = torch.pow(eigenvalues_stable, power)
    
    # 构建对角矩阵 Lambda^p
    lambda_powered_diag = torch.diag(eigenvalues_powered)
    
    # 重建: A^p = V @ Lambda^p @ V.T
    matrix_powered = eigenvectors @ lambda_powered_diag @ eigenvectors.T
    
    return matrix_powered

@torch.no_grad()
def EA_online_torch(x_tensor, R_tensor, i):
    """
    在线(增量)欧几里得对齐 (PyTorch GPU 版本)。
    x_tensor: (channels, time_samples) 的 GPU 张量
    R_tensor: 累积的协方差矩阵 (GPU 张量)
    """
    # 在 GPU 上计算协方差
    cov = torch.cov(x_tensor) 
    
    if i == 0:
        R_tensor = cov
    else:
        # 在 GPU 上进行增量更新
        R_tensor = (i * R_tensor + cov) / (i + 1)
        
    return R_tensor

@torch.no_grad()
def EA_torch(X_src_tensor, X_tar_tensor):
    """
    离线欧几里得对齐 (PyTorch GPU 版本)。
    X_src_tensor: (channels, N_src * time_samples) GPU 张量
    X_tar_tensor: (channels, N_tar * time_samples) GPU 张量
    """
    # 1. 在 GPU 上计算协方差
    cov_src = torch.cov(X_src_tensor)
    cov_tar = torch.cov(X_tar_tensor)
    
    # 2. 在 GPU 上计算矩阵的 -0.5 和 0.5 次幂
    R_src = matrix_power_torch(cov_src, -0.5) # (C, C)
    R_tar = matrix_power_torch(cov_tar, 0.5)  # (C, C)
    
    # 3. 在 GPU 上计算对齐矩阵 A = R_tar @ R_src
    A = R_tar @ R_src # (C, C)
    
    # 4. 在 GPU 上应用对齐
    # (C, C) @ (C, N_src*T) -> (C, N_src*T)
    X_src_aligned = A @ X_src_tensor
    # (C, C) @ (C, N_tar*T) -> (C, N_tar*T)
    X_tar_aligned = A @ X_tar_tensor
            
    return X_src_aligned, X_tar_aligned
  
#  添加针对“单个被试所有数据”的 GPU EA 函数
@torch.no_grad()
def EA_subject_gpu(x_tensor):
    """
    对单个被试的数据进行欧几里得对齐 (PyTorch GPU 版)。
    参数:
        x_tensor: (Trials, Channels, TimeSamples) 形状的 GPU 张量
    返回:
        aligned_tensor: 对齐后的 GPU 张量，形状不变
    """
    N, C, T = x_tensor.shape
    
    # 1. 中心化 (Centering)
    # 沿时间轴 (dim=2) 求均值
    mean = x_tensor.mean(dim=2, keepdim=True)
    x_centered = x_tensor - mean
    
    # 2. 计算协方差矩阵 (Covariance)
    # bmm: (N, C, T) @ (N, T, C) -> (N, C, C)
    # 这一步利用 GPU 并行计算所有 Trial 的协方差
    covs = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1)
    
    # 3. 计算参考矩阵 (Reference Matrix)
    # 所有 Trial 协方差的算术平均
    R = covs.mean(dim=0) # (C, C)
    
    # 4. 计算 R^(-0.5)
    sqrtRefEA = matrix_power_torch(R, -0.5) # (C, C)
    
    # 5. 应用对齐 (Alignment)
    # (1, C, C) @ (N, C, T) -> (N, C, T)
    # 利用广播机制一次性对齐所有 Trial
    x_aligned = torch.matmul(sqrtRefEA.unsqueeze(0), x_tensor)
    
    return x_aligned
# ==========================================================
# ==================== 新函数结束 ========================
# ==========================================================


