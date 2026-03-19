import argparse
import yaml
import sys
import logging
import time

from src.models.data_handler import DataStreamer
from src.models.data_analyzer import DataAnalyzer

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

        # TODO: ЭТАП 3: Подготовка данных и дообучение моделей

        status = True
    except Exception as e:
        logger.error(f"Ошибка в процессе Update: {e!r}")
        logger.exception(e)
        status = False
        
    execution_time = time.time() - start_time
    logger.info(f"Процесс UPDATE завершен за {execution_time:.2f} сек. Статус: {status}")
    return status

def run_inference(config, file_path):
    """
    Применение лучшей обученной модели к новым данным.
    """
    logger.info(f"Начат процесс INFERENCE для файла: {file_path}")
    if not file_path:
        logger.error("Для режима inference необходимо передать аргумент -file")
        return None
        
    try:
        # TODO: Загрузка модели через src.models.ml_models
        # TODO: Чтение file_path через pandas, predict, сохранение результата
        save_path = "data/external/predictions_output.csv"
        logger.info(f"Заглушка: предсказания сохранены в {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Ошибка в процессе Inference: {e}")
        return None

def run_summary(config):
    """
    Генерация отчета об изменении качества данных и моделей.
    """
    logger.info("Начат процесс SUMMARY (генерация отчетов)...")
    try:
        # TODO: Вызов View-модуля для агрегации метрик из storage/metrics/
        report_path = "storage/reports/summary_report.md"
        logger.info(f"Заглушка: отчет сгенерирован в {report_path}")
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