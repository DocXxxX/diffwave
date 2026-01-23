"""
参数表清洗脚本
处理监测参数表.csv的特殊格式（合并单元格导致的空值等问题）
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


def load_and_clean_params(params_csv_path: str) -> pd.DataFrame:
    """
    读取并清洗参数表
    
    Args:
        params_csv_path: 监测参数表CSV文件路径
        
    Returns:
        清洗后的DataFrame
    """
    # 读取CSV，跳过中文和单位行，保留英文列名
    df = pd.read_csv(params_csv_path, header=0, skiprows=[1, 2])
    
    # Event Level 填充：对关键列执行前向填充
    event_level_cols = ['Event_ID', 'Date', 'Q_max', 'Q_total', 'Hole_Num']
    for col in event_level_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    return df


def validate_params(df: pd.DataFrame) -> None:
    """
    数据校验（警告）
    检查关键参数是否存在空值
    """
    if 'Distance_R' in df.columns:
        missing_r = df['Distance_R'].isna().sum()
        if missing_r > 0:
            print(f"[警告] Distance_R (爆心距) 存在 {missing_r} 个空值，"
                  "必须在训练前补全这些距离数据，否则物理引导模块无法计算衰减。")
    
    if 'Elev_Diff' in df.columns:
        missing_h = df['Elev_Diff'].isna().sum()
        if missing_h > 0:
            print(f"[警告] Elev_Diff (高程差) 存在 {missing_h} 个空值。")


def build_params_dict(df: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """
    构建索引字典
    
    Args:
        df: 清洗后的DataFrame
        
    Returns:
        以 (Event_ID, Monitor_ID) 为Key，物理参数向量为Value的字典
    """
    params_dict = {}
    
    # 物理参数列：Q_max, Distance_R, Elev_Diff, Hole_Num, Delay_Int
    param_cols = ['Q_max', 'Distance_R', 'Elev_Diff', 'Hole_Num', 'Delay_Int']
    
    for _, row in df.iterrows():
        event_id = row['Event_ID']
        monitor_id = int(row['Monitor_ID'])
        key = (event_id, monitor_id)
        
        # 构建参数向量，缺失值使用默认值0
        params_vector = np.array([
            row.get(col, 0) if pd.notna(row.get(col, np.nan)) else 0.0
            for col in param_cols
        ], dtype=np.float32)
        
        params_dict[key] = params_vector
    
    return params_dict


def preprocess_params(params_csv_path: str) -> Dict[Tuple[str, int], np.ndarray]:
    """
    主函数：加载、清洗、验证并构建参数字典
    
    Args:
        params_csv_path: 监测参数表CSV文件路径
        
    Returns:
        参数索引字典
    """
    df = load_and_clean_params(params_csv_path)
    validate_params(df)
    params_dict = build_params_dict(df)
    
    print(f"[信息] 成功加载 {len(params_dict)} 条参数记录")
    return params_dict
