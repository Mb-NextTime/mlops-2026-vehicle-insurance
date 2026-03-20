import pandas as pd
import numpy as np
import os
import json
import logging
from mlxtend.frequent_patterns import apriori, association_rules

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, config):
        self.config = config
        self.metrics_dir = config['paths']['metrics']
        self.processed_dir = config['paths']['processed_data']

        self.max_missing = config['data_quality']['max_missing_threshold']
        self.min_support = config['data_quality']['apriori_min_support']
        self.min_confidence = config['data_quality']['apriori_min_confidence']

        os.makedirs(self.processed_dir, exist_ok=True)

    def assess_quality(self, df, batch_id):
        logger.info(f"Оценка качества данных для батча {batch_id}...")

        total_rows = len(df)
        dq_metrics = {
            "batch_id": batch_id,
            "missing_stats": {},
            "columns_to_drop": [],
            "outliers_count": {}
        }

        for col in df.columns:
            missing_pct = df[col].isna().sum() / total_rows
            dq_metrics["missing_stats"][col] = float(missing_pct)

            # Если пропусков больше порога, помечаем на удаление
            if missing_pct > self.max_missing:
                dq_metrics["columns_to_drop"].append(col)

        # Поиск аномалий
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            dq_metrics["outliers_count"][col] = int(outliers)

        dq_path = os.path.join(self.metrics_dir, f"batch_{batch_id}_dq.json")
        with open(dq_path, 'w') as f:
            json.dump(dq_metrics, f, indent=4)

        return dq_metrics

    def clean_data(self, df, dq_metrics):
        logger.info("Очистка данных на основе DQ метрик...")
        df_cleaned = df.copy()

        cols_to_drop = dq_metrics["columns_to_drop"]
        if cols_to_drop:
            logger.info(f"Удаление колонок из-за обилия пропусков: {cols_to_drop}")
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)

        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            df_cleaned = df_cleaned.drop_duplicates()
            logger.info(f"Удалено {duplicates} дубликатов.")

        return df_cleaned

    def generate_association_rules(self, df, batch_id):
        logger.info("Генерация ассоциативных правил (Apriori)...")

        ignore_cols = [
            self.config['data_collection']['target_column'],
            self.config['data_collection']['time_column'],
        ]

        cat_cols = [c for c in df.columns
                    if c not in ignore_cols
                    and 1 < df[c].nunique() < 10]

        if not cat_cols:
            logger.warning("Нет подходящих категориальных колонок для Apriori.")
            return None

        cat_cols = sorted(cat_cols, key=lambda x: df[x].nunique())[:5]

        df_bin = pd.get_dummies(df[cat_cols]).astype(bool)

        try:
            # Алгоритм Apriori
            frequent_itemsets = apriori(df_bin, min_support=self.min_support, use_colnames=True)

            if frequent_itemsets.empty:
                logger.warning("Apriori не нашел частых наборов (попробуйте снизить min_support).")
                return None

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)

            # Отбираем топ-10 правил по lift
            if not rules.empty:
                rules = rules.sort_values(by='lift', ascending=False).head(10)

                # Сохраняем правила
                rules_path = os.path.join(self.metrics_dir, f"batch_{batch_id}_rules.csv")
                rules.to_csv(rules_path, index=False)
                logger.info(f"Найдено {len(rules)} сильных правил. Сохранено в {rules_path}")
                return rules_path
            else:
                logger.info("Правила не преодолели порог confidence.")
                return None
        except Exception as e:
            logger.error(f"Ошибка при работе Apriori: {e}")
            return None

    def process_batch(self, raw_file_path):
        """Полный пайплайн анализа батча"""
        batch_id = os.path.basename(raw_file_path).split('_')[1].split('.')[0]
        df = pd.read_csv(raw_file_path)

        # Data Quality
        dq_metrics = self.assess_quality(df, batch_id)

        # Ассоциативные правила (на сырых данных до очистки)
        self.generate_association_rules(df, batch_id)

        # Очистка
        df_cleaned = self.clean_data(df, dq_metrics)

        # Сохранение очищенного батча
        processed_path = os.path.join(self.processed_dir, f"batch_{batch_id}_clean.csv")
        df_cleaned.to_csv(processed_path, index=False)
        logger.info(f"Очищенный батч сохранен в {processed_path}")

        return processed_path
