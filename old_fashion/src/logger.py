import logging
import sys
import os
from src.config import LOGS_DIR

class Colors:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"

def setup_logger(name: str, log_filename: str = "app.log", level=logging.INFO):
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Формат для файла (Полный)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-15s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 2. Формат для консоли (Краткий + Цветной)
    # Мы добавляем цвета прямо в строку формата
    console_fmt = logging.Formatter(
        fmt=f"{Colors.BLUE}%(asctime)s{Colors.RESET} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Хендлер консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Хендлер файла
    file_path = os.path.join(LOGS_DIR, log_filename)
    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
