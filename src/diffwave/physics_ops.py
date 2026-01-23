"""
可微物理算子模块
实现用于爆破振动分析的可微分算子
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict


class DifferentiableWPT(nn.Module):
    """可微小波包变换"""
    
    def __init__(self, 
                 sample_rate: int = 8000,
                 wavelet: str = 'db4',
                 level: int = 6):
        """
        Args:
            sample_rate: 采样率
            wavelet: 小波基类型
            level: 分解层数 (8kHz下建议6-7层)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.level = level
        
        # 初始化小波滤波器
        self._init_filters(wavelet)
    
    def _init_filters(self, wavelet: str):
        """初始化小波滤波器"""
        # db4小波滤波器系数
        if wavelet == 'db4':
            lo = np.array([
                -0.010597401784997,  0.032883011666983,
                 0.030841381835987, -0.187034811718881,
                -0.027983769416984,  0.630880767929590,
                 0.714846570552542,  0.230377813308855
            ])
            hi = np.array([
                -0.230377813308855,  0.714846570552542,
                -0.630880767929590, -0.027983769416984,
                 0.187034811718881,  0.030841381835987,
                -0.032883011666983, -0.010597401784997
            ])
        else:
            # 默认使用Haar小波
            lo = np.array([0.7071067811865476, 0.7071067811865476])
            hi = np.array([-0.7071067811865476, 0.7071067811865476])
        
        # 注册为buffer，注意维度 [Out, In, Kernel] -> [1, 1, K]
        # 对于conv1d: weight shape [out_channels, in_channels, kernel_size]
        # 这里是单通道处理
        self.register_buffer('lo_d', torch.tensor(lo, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('hi_d', torch.tensor(hi, dtype=torch.float32).view(1, 1, -1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行WPT分解并计算各频带能量
        
        Args:
            x: [B, T] 输入波形
            
        Returns:
            [B, 2^level] 各频带能量
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        
        coeffs = [x]
        
        for _ in range(self.level):
            new_coeffs = []
            for c in coeffs:
                # 需处理padding以保持尺寸或允许尺寸减半
                # padding='same' in conv1d requires specific pytorch versions or manual padding
                # 这里简单处理：使用 reflect padding
                # kernel size
                k = self.lo_d.shape[-1]
                pad = k // 2
                
                # Low frequency
                # 暂时使用 conv1d stride 2 进行下采样
                c_pad = torch.nn.functional.pad(c, (pad, pad), mode='reflect')
                
                lo = torch.nn.functional.conv1d(c_pad, self.lo_d)
                # Downsample
                lo = lo[:, :, ::2]
                
                hi = torch.nn.functional.conv1d(c_pad, self.hi_d)
                # Downsample
                hi = hi[:, :, ::2]
                
                new_coeffs.extend([lo, hi])
            coeffs = new_coeffs
        
        # 计算各频带能量
        # coeffs is list of [B, 1, T'] tensors
        # Energy = sum(x^2)
        energies = torch.stack([
            torch.sum(c ** 2, dim=-1).squeeze(1) 
            for c in coeffs
        ], dim=1)
        
        return energies
    
    def get_frequency_bands(self) -> List[Tuple[float, float]]:
        """获取各频带对应的频率范围"""
        nyquist = self.sample_rate / 2
        n_bands = 2 ** self.level
        band_width = nyquist / n_bands
        
        # 注意：WPT频带顺序不仅是简单的低到高，由于滤波器组的特性，存在频率倒置
        # 这里简化处理，假设是自然序（实际需要基于格雷码排序）
        # 暂时返回线性划分
        return [(i * band_width, (i + 1) * band_width) for i in range(n_bands)]


class PhysicsLoss(nn.Module):
    """物理引导损失"""
    
    def __init__(self, 
                 sample_rate: int = 8000,
                 wpt_level: int = 6,
                 lambda_wpt: float = 0.1):
        """
        Args:
            sample_rate: 采样率
            wpt_level: WPT分解层数
            lambda_wpt: WPT能量损失权重
        """
        super().__init__()
        self.wpt = DifferentiableWPT(sample_rate, level=wpt_level)
        self.lambda_wpt = lambda_wpt
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                physical_params: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算物理引导损失
        
        Args:
            pred: [B, T] 预测波形
            target: [B, T] 目标波形
            physical_params: [B, 5] 物理参数（用于自适应权重，预留）
            
        Returns:
            (total_loss, losses_dict)
        """
        losses = {}
        
        # 基础重建损失 (L1)
        l1_loss = torch.mean(torch.abs(pred - target))
        losses['l1'] = l1_loss
        
        # WPT能量分布损失
        pred_energy = self.wpt(pred)
        target_energy = self.wpt(target)
        
        # 归一化能量分布 (避免因幅值差异导致的巨大误差，关注频谱形状)
        pred_energy_norm = pred_energy / (pred_energy.sum(dim=1, keepdim=True) + 1e-8)
        target_energy_norm = target_energy / (target_energy.sum(dim=1, keepdim=True) + 1e-8)
        
        wpt_loss = torch.mean((pred_energy_norm - target_energy_norm) ** 2)
        losses['wpt'] = wpt_loss
        
        # 总损失
        total_loss = l1_loss + self.lambda_wpt * wpt_loss
        losses['total'] = total_loss
        
        return total_loss, losses
