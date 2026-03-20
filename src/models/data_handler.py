import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStreamer:
    def __init__(self, config):
        self.config = config
        self.initial_file = config['paths']['initial_data']
        self.raw_dir = config['paths']['raw_data']
        self.metrics_dir = config['paths']['metrics']
        self.state_file = config['paths']['stream_state']

        self.batch_size = config['data_collection']['batch_size']
        self.time_col = config['data_collection']['time_column']
        self.miss_rate = config['data_collection']['missing_value_injection_rate']

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _get_current_offset(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                return state.get('offset', 0), state.get('batch_id', 0)
        return 0, 0

    def _save_current_offset(self, offset, batch_id):
        with open(self.state_file, 'w') as f:
            json.dump({'offset': offset, 'batch_id': batch_id}, f)

    def _inject_missing_values(self, df):
        logger.info(f"Добавление {self.miss_rate*100}% случайных пропусков...")
        # Выбираем только признаки (не трогаем таргет и время)
        target = self.config['data_collection']['target_column']
        cols_to_modify = [c for c in df.columns if c not in [target, self.time_col]]

        for col in cols_to_modify:
            # Маска для замены на NaN
            mask = np.random.rand(len(df)) < self.miss_rate
            df.loc[mask, col] = np.nan
        return df

    def _calculate_meta_parameters(self, df, batch_id):
        num_cols = df.select_dtypes(include=[np.number]).columns

        meta = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_values_total": int(df.isna().sum().sum()),
            "numerical_stats": {}
        }

        for col in num_cols:
            meta["numerical_stats"][col] = {
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "variance": float(df[col].var()) if not pd.isna(df[col].var()) else None
            }

        meta_path = os.path.join(self.metrics_dir, f"batch_{batch_id}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)
        logger.info(f"Метапараметры батча сохранены в {meta_path}")

    def get_next_batch(self):
        """Основной метод: получение нового батча."""
        offset, batch_id = self._get_current_offset()

        try:
            logger.info(f"Чтение исходных данных из {self.initial_file}")
            df_full = pd.read_csv(self.initial_file)

            if self.time_col in df_full.columns:
                df_full[self.time_col] = pd.to_datetime(df_full[self.time_col], errors='coerce', format='mixed')
                df_full = df_full.sort_values(by=self.time_col)

            if offset >= len(df_full):
                logger.warning("Конец потока: новые данные закончились.")
                return None

            df_batch = df_full.iloc[offset : offset + self.batch_size].copy()
            batch_id += 1

            df_batch = self._inject_missing_values(df_batch)

            self._calculate_meta_parameters(df_batch, batch_id)

            batch_filename = os.path.join(self.raw_dir, f"batch_{batch_id}.csv")
            df_batch.to_csv(batch_filename, index=False)
            logger.info(f"Сформирован и сохранен {batch_filename} ({len(df_batch)} строк).")

            self._save_current_offset(offset + len(df_batch), batch_id)

            return batch_filename

        except Exception as e:
            logger.error(f"Ошибка при получении батча: {e}")
            raise
