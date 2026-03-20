import pandas as pd
import numpy as np
import os
import json
import logging
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models_dir = config['paths']['models']
        self.metrics_dir = config['paths']['metrics']

        self.target_col = config['data_collection']['target_column']
        self.time_col = config['data_collection']['time_column']
        self.test_size = config['training']['test_size']
        self.random_state = config['training']['random_state']

        os.makedirs(self.models_dir, exist_ok=True)

        self.prep_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        self.rf_path = os.path.join(self.models_dir, 'model_rf.pkl')
        self.mlp_path = os.path.join(self.models_dir, 'model_mlp.pkl')

    def _build_preprocessor(self, X):
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ])

        return preprocessor

    def train_or_update(self, processed_batch_path):
        batch_id = os.path.basename(processed_batch_path).split('_')[1]
        logger.info(f"Начало ML пайплайна для батча {batch_id}...")

        df = pd.read_csv(processed_batch_path)

        if self.target_col not in df.columns:
            logger.error(f"Целевая переменная {self.target_col} не найдена!")
            return False

        drop_cols = [self.target_col]
        if self.time_col in df.columns:
            drop_cols.append(self.time_col)

        X = df.drop(columns=drop_cols)
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        if os.path.exists(self.prep_path):
            preprocessor = joblib.load(self.prep_path)
            X_train_prep = preprocessor.transform(X_train)
            logger.info("Загружен существующий препроцессор.")
        else:
            preprocessor = self._build_preprocessor(X_train)
            X_train_prep = preprocessor.fit_transform(X_train)
            joblib.dump(preprocessor, self.prep_path)
            logger.info("Обучен и сохранен новый препроцессор.")

        X_test_prep = preprocessor.transform(X_test)

        # Получаем все возможные классы (нужно для partial_fit нейросети)
        all_classes = self.config['training']['targets']

        # partial_fit с warm_start у MLPClassifier плохо работает вместе
        # если в батче окажутся не все классы — выкинет ошибку
        # поэтому добавляем фиктивные строки, дополняющие недостающие классы
        # в контексте батча в 5000, лишние 3-4 строки не сыграют особой роли
        missing_classes = np.setdiff1d(all_classes, np.unique(y_train))
        if len(missing_classes) > 0:
            logger.info(f"В батче отсутствуют классы: {missing_classes}. Добавляем фиктивные строки.")
            dummy_X = X_train_prep[:len(missing_classes)]
            dummy_y = pd.Series(missing_classes)
            X_train_prep = np.vstack([X_train_prep, dummy_X])
            y_train = pd.concat([y_train, dummy_y], ignore_index=True)

        if os.path.exists(self.rf_path):
            rf_model = joblib.load(self.rf_path)
            # Дообучение леса: добавляем 10 деревьев к ансамблю (warm_start)
            rf_model.n_estimators += 10
            rf_model.fit(X_train_prep, y_train)
            logger.info(f"RandomForest Дообучен. Всего деревьев: {rf_model.n_estimators}")
        else:
            rf_model = RandomForestClassifier(n_estimators=50, warm_start=True, random_state=self.random_state)
            rf_model.fit(X_train_prep, y_train)
            logger.info("RandomForest обучен с нуля.")
        joblib.dump(rf_model, self.rf_path)

        if os.path.exists(self.mlp_path):
            mlp_model = joblib.load(self.mlp_path)
            mlp_model.partial_fit(X_train_prep, y_train)
            logger.info("Нейросеть (MLP) ДОобучена (partial_fit).")
        else:
            mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=False, random_state=self.random_state)
            mlp_model.partial_fit(X_train_prep, y_train, classes=all_classes)
            logger.info("Нейросеть (MLP) обучена с нуля.")
        joblib.dump(mlp_model, self.mlp_path)

        rf_preds = rf_model.predict(X_test_prep)
        mlp_preds = mlp_model.predict(X_test_prep)

        metrics = {
            "batch_id": batch_id,
            "test_size": len(y_test),
            "models": {
                "RandomForest": {
                    "accuracy": float(accuracy_score(y_test, rf_preds)),
                    "f1_macro": float(f1_score(y_test, rf_preds, average='macro'))
                },
                "MLP_NeuralNet": {
                    "accuracy": float(accuracy_score(y_test, mlp_preds)),
                    "f1_macro": float(f1_score(y_test, mlp_preds, average='macro'))
                }
            }
        }

        metrics_file = os.path.join(self.metrics_dir, f"batch_{batch_id}_ml_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Метрики сохранены в {metrics_file}")
        logger.info(f"RF Acc: {metrics['models']['RandomForest']['accuracy']:.4f} | MLP Acc: {metrics['models']['MLP_NeuralNet']['accuracy']:.4f}")

        return True