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

        self.prep_path = 'preprocessor.pkl'
        self.registry_path = os.path.join(self.models_dir, 'model_registry.json')

        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "latest": {},
                "best_model": {
                    "name": None,
                    "path": None,
                    "accuracy": 0.0
                },
                "history": []
            }

    def _save_registry(self):
        """Сохранение состояния реестра моделей"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4, ensure_ascii=False)

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

        if "preprocessor" in self.registry["latest"]:
            preprocessor = joblib.load(os.path.join(self.models_dir, self.registry["latest"]["preprocessor"]))
            X_train_prep = preprocessor.transform(X_train)
            logger.info("Загружен существующий препроцессор.")
        else:
            preprocessor = self._build_preprocessor(X_train)
            X_train_prep = preprocessor.fit_transform(X_train)
            joblib.dump(preprocessor, os.path.join(self.models_dir, self.prep_path))
            self.registry["latest"]["preprocessor"] = self.prep_path
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

        rf_file = f"model_rf_batch_{batch_id}.pkl"
        mlp_file = f"model_mlp_batch_{batch_id}.pkl"

        # RandomForest
        if "RandomForest" in self.registry["latest"]:
            old_rf_path = os.path.join(self.models_dir, self.registry["latest"]["RandomForest"])
            rf_model = joblib.load(old_rf_path)
            rf_model.n_estimators += 10 # Дообучение
            logger.info(f"Загружена предыдущая версия RF. Увеличиваем ансамбль до {rf_model.n_estimators} деревьев.")
        else:
            rf_model = RandomForestClassifier(n_estimators=50, warm_start=True, random_state=self.random_state)
            logger.info("Инициализирован новый RandomForest.")

        rf_model.fit(X_train_prep, y_train)
        joblib.dump(rf_model, os.path.join(self.models_dir, rf_file))

        # MLP Neural Network
        if "MLP_NeuralNet" in self.registry["latest"]:
            old_mlp_path = os.path.join(self.models_dir, self.registry["latest"]["MLP_NeuralNet"])
            mlp_model = joblib.load(old_mlp_path)
            logger.info("Загружена предыдущая версия MLP.")
        else:
            mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, warm_start=False, random_state=self.random_state)
            logger.info("Инициализирован MLP с нуля.")

        mlp_model.partial_fit(X_train_prep, y_train, classes=all_classes)
        joblib.dump(mlp_model, os.path.join(self.models_dir, mlp_file))

        rf_preds = rf_model.predict(X_test_prep)
        mlp_preds = mlp_model.predict(X_test_prep)

        rf_acc = float(accuracy_score(y_test, rf_preds))
        mlp_acc = float(accuracy_score(y_test, mlp_preds))

        self.registry["latest"]["RandomForest"] = rf_file
        self.registry["latest"]["MLP_NeuralNet"] = mlp_file

        if rf_acc > self.registry["best_model"]["accuracy"]:
            self.registry["best_model"] = {"name": "RandomForest", "path": rf_file, "accuracy": rf_acc}
            logger.info(f"Новая лучшая модель: RandomForest из батча {batch_id} (Acc: {rf_acc:.4f})")

        if mlp_acc > self.registry["best_model"]["accuracy"]:
            self.registry["best_model"] = {"name": "MLP_NeuralNet", "path": mlp_file, "accuracy": mlp_acc}
            logger.info(f"Новая лучшая модель: MLP из батча {batch_id} (Acc: {mlp_acc:.4f})")

        metrics = {
            "batch_id": batch_id,
            "test_size": len(y_test),
            "models": {
                "RandomForest": {"accuracy": rf_acc, "f1_macro": float(f1_score(y_test, rf_preds, average='macro'))},
                "MLP_NeuralNet": {"accuracy": mlp_acc, "f1_macro": float(f1_score(y_test, mlp_preds, average='macro'))}
            }
        }
        self.registry["history"].append(metrics)
        self._save_registry()

        metrics_file = os.path.join(self.metrics_dir, f"batch_{batch_id}_ml_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"RF Acc: {rf_acc:.4f} | MLP Acc: {mlp_acc:.4f}")
        return True
