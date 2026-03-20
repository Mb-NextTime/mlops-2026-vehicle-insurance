import os
import shutil
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def reset_environment():
    config = load_config()
    dirs_to_clean = [
        config['paths']['raw_data'],
        config['paths']['processed_data'],
        config['paths']['metrics'],
        config['paths']['reports'],
        config['paths']['models']
    ]

    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            logging.info(f"Удалена директория: {d}")
        os.makedirs(d, exist_ok=True)
        logging.info(f"Создана чистая директория: {d}")


    state_file = config['paths']['stream_state']
    if os.path.exists(state_file):
        os.remove(state_file)
        logging.info(f"Удален файл состояния: {state_file}")

    logging.info("Система успешно сброшена к исходному состоянию!")

if __name__ == "__main__":
    reset_environment()
