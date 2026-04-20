import os
import pytest
import yaml

def test_config_exists():
    assert os.path.exists("config.yaml"), "Конфигурационный файл не найден!"
    
def test_config_validity():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "paths" in config, "В конфиге отсутствует секция paths"
    assert "data_collection" in config, "В конфиге отсутствует секция data_collection"
    assert config["data_collection"]["batch_size"] > 0, "Размер батча должен быть > 0"

def test_initial_data_exists():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    initial_data_path = config["paths"]["initial_data"]
    assert os.path.exists(initial_data_path), f"Исходный датасет не найден по пути {initial_data_path}"
