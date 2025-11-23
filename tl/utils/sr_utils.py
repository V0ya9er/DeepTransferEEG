# -*- coding: utf-8 -*-
import numpy as np
import torch
import math

def apply_bistable_sr(signal_data, dt, noise_intensity, a=1.0, b=1.0):
    """
    应用经典双稳态随机共振模型。

    参数:
    signal_data (np.array): 输入信号，形状应为 (1, channels, time_samples) 
                            或 (channels, time_samples)。
    dt (float): 时间步长, 等于 1 / 采样率。
    noise_intensity (float): 噪声强度 (D), 这是需要调优的关键参数。
    a, b (float): 双稳态势 U(x) = -a/2 * x^2 + b/4 * x^4 的参数。

    返回:
    np.array: SR系统输出, 形状与输入相同。
    """
    original_shape = signal_data.shape
    
    # SR模型在 (channels, time_samples) 的2D数据上运行
    if len(original_shape) >= 3 and original_shape[0] == 1:
        # (1, C, T) -> (C, T)
        signal_data_2d = signal_data.reshape((original_shape[-2], original_shape[-1]))
    elif len(original_shape) == 2:
        # (C, T)
        signal_data_2d = signal_data
    else:
        # 如果数据格式不符, 抛出错误
        raise ValueError(f"随机共振(SR)只支持 (1, C, T) 或 (C, T) 的形状, 但收到了: {original_shape}")

    num_channels, num_samples = signal_data_2d.shape
    
    # 归一化输入信号 (按通道)
    signal_mean = np.mean(signal_data_2d, axis=1, keepdims=True)
    signal_std = np.std(signal_data_2d, axis=1, keepdims=True)
    # 避免除以零
    normalized_signal = (signal_data_2d - signal_mean) / (signal_std + 1e-9)
    
    output_signal = np.zeros_like(normalized_signal)
    x = np.zeros(num_channels)  # 每个通道的初始状态 x(t=0)
    
    # 预计算噪声缩放因子
    noise_scale = np.sqrt(2 * noise_intensity * dt)
    
    # Euler-Maruyama 积分
    for t in range(num_samples):
        # dx = (a*x - b*x^3 + S(t)) * dt + sqrt(2*D*dt) * W(t)
        deterministic_force = (a * x - b * np.power(x, 3)) + normalized_signal[:, t]
        stochastic_force = np.random.normal(0.0, 1.0, num_channels) * noise_scale
        
        # 更新状态
        x = x + deterministic_force * dt + stochastic_force
        output_signal[:, t] = x
        
    # 将输出重塑为原始形状, e.g., (1, C, T)
    return output_signal.reshape(original_shape)
  
  
@torch.jit.script
def apply_bistable_sr_torch(
    signal_data_tensor: torch.Tensor,  # <--- 显式标注为 Tensor
    dt: float,                         # <--- 显式标注为 float
    noise_intensity: float,            # <--- 显式标注为 float
    a: float = 1.0,                    # <--- 显式标注为 float，解决默认值冲突
    b: float = 1.0                     # <--- 显式标注为 float
) -> torch.Tensor:                     # <--- (可选) 标注返回类型
    """
    应用经典双稳态随机共振模型。(PyTorch GPU 版本)

    参数:
    signal_data_tensor (torch.Tensor): 输入信号张量，应在目标设备上 (e.g., GPU)。
                                     形状应为 (1, channels, time_samples) 
                                     或 (channels, time_samples)。
    dt (float): 时间步长, 等于 1 / 采样率。
    noise_intensity (float): 噪声强度 (D)。
    a, b (float): 双稳态势 U(x) = -a/2 * x^2 + b/4 * x^4 的参数。

    返回:
    torch.Tensor: SR系统输出, 形状与输入相同, 位于同一设备。
    """
    
    # 从输入张量获取设备和数据类型
    device = signal_data_tensor.device
    dtype = signal_data_tensor.dtype
    
    original_shape = signal_data_tensor.shape
    
    # SR模型在 (channels, time_samples) 的2D数据上运行
    if len(original_shape) == 3 and original_shape[0] == 1:
        # (1, C, T) -> (C, T)
        signal_data_2d = signal_data_tensor.squeeze(0)
    elif len(original_shape) == 2:
        # (C, T)
        signal_data_2d = signal_data_tensor
    else:
        # 如果数据格式不符, 抛出错误
        raise ValueError(f"随机共振(SR)只支持 (1, C, T) 或 (C, T) 的形状, 但收到了: {original_shape}")

    num_channels, num_samples = signal_data_2d.shape
    
    # 1. 归一化输入信号 (PyTorch)
    signal_mean = torch.mean(signal_data_tensor, dim=1, keepdim=True)
    signal_std = torch.std(signal_data_tensor, dim=1, keepdim=True)
    normalized_signal = (signal_data_tensor - signal_mean) / (signal_std + 1e-9)
    
    # 2. 初始化 (PyTorch), 确保在新张量上指定 device 和 dtype
    output_signal = torch.zeros_like(normalized_signal)
    x = torch.zeros(signal_data_tensor.shape[0], device=signal_data_tensor.device, dtype=signal_data_tensor.dtype)  # 每个通道的初始状态 x(t=0)
    
    # 3. 预计算噪声缩放因子 (PyTorch)
    # 将标量值转换为 tensor，以便在 GPU 上进行后续计算
    noise_scale = math.sqrt(2 * noise_intensity * dt)
    
    # Euler-Maruyama 积分
    for t in range(signal_data_tensor.shape[1]):
        
        # 确定性力
        deterministic_force = (a * x - b * torch.pow(x, 3)) + normalized_signal[:, t]
        
        # 随机力
        # noise_scale 是 float，直接乘即可
        stochastic_force = torch.randn_like(x) * noise_scale
        
        # 更新状态
        x = x + deterministic_force * dt + stochastic_force
        output_signal[:, t] = x
        
    return output_signal.reshape(signal_data_tensor.shape)
  
# apply_bistable_sr_opt = torch.compile(apply_bistable_sr_torch)
# Linux上才能能用这一行
  
def apply_bistable_sr_batch(signal_data, dt, noise_intensity, a=1.0, b=1.0):
    """
    应用经典双稳态随机共振模型 (Batch 批处理版本)。

    参数:
    signal_data (np.array): 输入信号, 形状 (N, C, T)。
                            N=试验数, C=通道数, T=时间点数。
    dt (float): 时间步长, 等于 1 / 采样率。
    noise_intensity (float): 噪声强度 (D)。
    a, b (float): 势函数参数 U(x) = -a/2 * x^2 + b/4 * x^4。

    返回:
    np.array: SR系统输出, 形状 (N, C, T)。
    """
    if len(signal_data.shape) != 3:
        raise ValueError(f"SR Batch (N, C, T) 形状, 但收到: {signal_data.shape}")

    N, C, T = signal_data.shape
    
    # 归一化 (N, C, T) -> (N, C, 1)
    signal_mean = np.mean(signal_data, axis=2, keepdims=True)
    signal_std = np.std(signal_data, axis=2, keepdims=True)
    # 避免除以零
    normalized_signal = (signal_data - signal_mean) / (signal_std + 1e-9)
    
    output_signal = np.zeros_like(normalized_signal)
    x = np.zeros((N, C))  # 初始状态 (N, C)
    
    # 预计算噪声缩放因子
    noise_scale = np.sqrt(2 * noise_intensity * dt)
    
    # Euler-Maruyama 积分 (按时间步迭代)
    for t in range(T):
        # S(t) 形状 (N, C)
        S_t = normalized_signal[:, :, t]
        
        # 确定性力: (a*x - b*x^3 + S(t))
        deterministic_force = (a * x - b * np.power(x, 3)) + S_t
        
        # 随机力: W(t) * scale 形状 (N, C)
        stochastic_force = np.random.normal(0.0, 1.0, (N, C)) * noise_scale
        
        # 更新状态 x
        x = x + deterministic_force * dt + stochastic_force
        output_signal[:, :, t] = x
        
    return output_signal
  
def apply_msr_to_channel(signal_1d, a, b, D, dt):
    """
    对单个通道（1D信号）应用双稳态随机共振。
    
    参数:
    signal_1d (np.array): 输入的1D EEG信号 (长度为 time_sample_num)。
    a (float): 双稳态势参数 a。
    b (float): 双稳态势参数 b。
    D (float): 噪声强度。
    dt (float): 数值积分的时间步长 (例如 1 / 采样率)。
    
    返回:
    np.array: 经过MSR处理后的1D信号。
    """
    
    # 初始化输出信号数组
    x_out = np.zeros_like(signal_1d)
    
    # 设置初始条件
    x_out[0] = 0.0  
    
    # 预先计算噪声项的系数
    # n(t) * dt 约等于 sqrt(2 * D * dt) * np.random.randn()
    noise_coeff = np.sqrt(2 * D * dt)
    
    # 使用欧拉-丸山法进行迭代
    for i in range(len(signal_1d) - 1):
        # 朗之万方程: dx = (ax - bx^3 + s(t)) * dt + sqrt(2*D*dt) * W_i
        
        # 确定性部分 (漂移项)
        drift = (a * x_out[i] - b * (x_out[i]**3) + signal_1d[i]) * dt
        
        # 随机部分 (扩散项)
        noise = noise_coeff * np.random.randn()
        
        # 更新下一个时间点的值
        x_out[i+1] = x_out[i] + drift + noise
        
    return x_out
  
