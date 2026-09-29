import logging
import os

# settings
LOG_LEVEL = logging.DEBUG
LOG_DIR = "logs"
LOG_FILE = "app.log"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# set root logger configuration
def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
