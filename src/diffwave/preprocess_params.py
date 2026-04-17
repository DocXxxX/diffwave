"""
参数表清洗脚本
处理监测参数表.csv的特殊格式（合并单元格导致的空值等问题）
"""
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


BLAST_PARAM_COLUMNS = [
  'Q_max',
  'Q_total',
  'Hole_Num',
  'Delay_hole',
  'Delay_row',
  'Hole_Diameter',
  'Distance_R',
  'Elev_Diff',
]

EVENT_LEVEL_COLUMNS = [
  'Event_ID',
  'Date',
  'Q_max',
  'Q_total',
  'Hole_Num',
  'Delay_hole',
  'Delay_row',
  'Hole_Diameter',
]


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
  df.columns = [col.strip() for col in df.columns]
  df = df.dropna(how='all').copy()

  # Event Level 填充：对关键列执行前向填充
  for col in EVENT_LEVEL_COLUMNS:
    if col in df.columns:
      df[col] = df[col].ffill()

  if 'Monitor_ID' in df.columns:
    df = df[df['Monitor_ID'].notna()].copy()
    df['Monitor_ID'] = pd.to_numeric(df['Monitor_ID'], errors='coerce')
    df = df[df['Monitor_ID'].notna()].copy()
    df['Monitor_ID'] = df['Monitor_ID'].astype(int)

  for col in BLAST_PARAM_COLUMNS:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors='coerce')

  df['DateKey'] = df['Event_ID'].apply(_date_key_from_event_id)
  return df


def validate_params(df: pd.DataFrame) -> None:
  """
  数据校验（警告）
  检查关键参数是否存在空值
  """
  missing_cols = [col for col in BLAST_PARAM_COLUMNS if col not in df.columns]
  if missing_cols:
    print(f"[WARN] Parameter table is missing columns: {missing_cols}")

  for col in BLAST_PARAM_COLUMNS:
    if col in df.columns:
      missing = int(df[col].isna().sum())
      if missing > 0:
        print(f"[WARN] {col} has {missing} missing values; related samples will be skipped.")


def _date_key_from_event_id(event_id: str) -> str:
  match = re.match(r'^BL(\d{8})', str(event_id))
  return match.group(1) if match else ''


def build_unique_param_index(df: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
  index = {}
  for key, group in df.groupby(['DateKey', 'Monitor_ID'], dropna=False):
    if len(group) == 1:
      index[key] = group.iloc[0]
  return index


def build_params_dict(df: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
  """
  构建索引字典

  Args:
      df: 清洗后的DataFrame

  Returns:
      以 (Event_ID, Monitor_ID) 为Key，物理参数向量为Value的字典
  """
  params_dict = {}

  for _, row in df.iterrows():
    if any(pd.isna(row.get(col, np.nan)) for col in BLAST_PARAM_COLUMNS):
      continue
    event_id = row['Event_ID']
    monitor_id = int(row['Monitor_ID'])
    key = (event_id, monitor_id)
    params_dict[key] = row[BLAST_PARAM_COLUMNS].to_numpy(dtype=np.float32)

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

  print(f"[INFO] Loaded {len(params_dict)} parameter records.")
  return params_dict
