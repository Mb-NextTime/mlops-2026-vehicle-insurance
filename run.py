import argparse
import yaml
import sys
import logging
import time
import os

from src.models.data_handler import DataStreamer
from src.models.data_analyzer import DataAnalyzer
from src.models.ml_model import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("storage/reports/system.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_update(config):
    """
    Эмуляция получения нового батча данных, проверка Data Quality,
    дообучение моделей и сохранение метрик.
    """
    logger.info("Начат процесс UPDATE (сбор данных, анализ, дообучение)...")
    start_time = time.time()

    try:
        # Сбор данных
        streamer = DataStreamer(config)
        new_batch_path = streamer.get_next_batch()

        if not new_batch_path:
            logger.info("Нет новых данных для обработки.")
            return False

        # Анализ данных (Data Quality + EDA)
        analyzer = DataAnalyzer(config)
        clean_batch_path = analyzer.process_batch(new_batch_path)

        model_manager = ModelManager(config)
        ml_status = model_manager.train_or_update(clean_batch_path)

        status = ml_status
    except Exception as e:
        logger.error(f"Ошибка в процессе Update: {e!r}")
        logger.exception(e)
        status = False
        
    execution_time = time.time() - start_time
    logger.info(f"Процесс UPDATE завершен за {execution_time:.2f} сек. Статус: {status}")
    return status

def run_inference(config, file_path):
    """
    Применение лучшей обученной модели к внешним данным.
    """
    logger.info(f"Начат процесс INFERENCE для файла: {file_path}")
    if not file_path or not os.path.exists(file_path):
        logger.error("Для режима inference необходимо передать корректный путь -file")
        return None
        
    try:
        import pandas as pd
        import joblib
        import json

        registry_path = os.path.join(config['paths']['models'], 'model_registry.json')
        if not os.path.exists(registry_path):
            logger.error("Реестр моделей не найден, сначала запустите режим update.")
            return None

        with open(registry_path, 'r') as f:
            registry = json.load(f)

        best_model_info = registry.get("best_model", {})
        if not best_model_info.get("path"):
            logger.error("В реестре нет лучшей модели.")
            return None

        logger.info(f"Выбрана лучшая модель: {best_model_info['name']} (Acc: {best_model_info['accuracy']:.4f}) из файла {best_model_info['path']}")

        prep_path = os.path.join(config['paths']['models'], registry["latest"]["preprocessor"])
        model_path = os.path.join(config['paths']['models'], best_model_info["path"])

        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)

        df = pd.read_csv(file_path)
        # убираем таргет
        target_col = config['data_collection']['target_column']
        if target_col in df.columns:
            df_features = df.drop(columns=[target_col])
        else:
            df_features = df.copy()

        X_prep = preprocessor.transform(df_features)

        predictions = model.predict(X_prep)

        df['predict'] = predictions

        os.makedirs(config['paths']['external_data'], exist_ok=True)
        save_path = os.path.join(config['paths']['external_data'], "predictions_output.csv")
        df.to_csv(save_path, index=False)

        logger.info(f"Прогнозы успешно сохранены в {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Ошибка в процессе Inference: {e}")
        return None

def run_summary(config):
    """
    Генерация Markdown отчета об изменении метрик (Во времени).
    """
    logger.info("Начат процесс SUMMARY (генерация отчетов)...")
    try:
        import os
        import json
        import pandas as pd

        metrics_dir = config['paths']['metrics']
        reports_dir = config['paths']['reports']
        os.makedirs(reports_dir, exist_ok=True)

        # Собираем данные по всем батчам
        summary_data = []
        for file in sorted(os.listdir(metrics_dir)):
            if file.endswith("_ml_metrics.json"):
                with open(os.path.join(metrics_dir, file), 'r') as f:
                    data = json.load(f)
                    summary_data.append({
                        "Batch": data["batch_id"],
                        "RF_Accuracy": data["models"]["RandomForest"]["accuracy"],
                        "MLP_Accuracy": data["models"]["MLP_NeuralNet"]["accuracy"]
                    })

        if not summary_data:
            logger.warning("Нет метрик для генерации отчета.")
            return None

        df_summary = pd.DataFrame(summary_data)

        # Генерируем Markdown файл
        report_path = os.path.join(reports_dir, "summary_report.md")
        with open(report_path, "w") as f:
            f.write("# Отчет о мониторинге ML-Системы (Summary)\n\n")
            f.write("## Динамика качества моделей (Accuracy)\n\n")
            f.write(df_summary.to_markdown(index=False))
            f.write("\n\n*Отчет сгенерирован автоматически.*\n")

        logger.info(f"Отчет сгенерирован в {report_path}")
        return report_path

    except Exception as e:
        logger.error(f"Ошибка в процессе Summary: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps MVP Pipeline CLI")
    parser.add_argument("-mode", type=str, required=True, choices=["inference", "update", "summary"],
                        help="Режим работы программы")
    parser.add_argument("-file", type=str, required=False,
                        help="Путь к файлу с внешними данными (только для режима inference)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    
    if args.mode == "update":
        result = run_update(config)
        print(f"Update Result: {result}")
        
    elif args.mode == "inference":
        result = run_inference(config, args.file)
        print(f"Inference Result (Saved to): {result}")
        
    elif args.mode == "summary":
        result = run_summary(config)
        print(f"Summary Report Path: {result}")

if __name__ == "__main__":
    main()
