import logging
import os
import sys
from datetime import datetime


def setup_logging(input_file, task_id, reviser_size):
    
    # Extract base name from input file and create log file names with timestamp
    base_name = os.path.basename(input_file).split('_')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if task_id == -1:
        save_path = os.path.join('runs', base_name, str(reviser_size))
    else:
        save_path = os.path.join('runs', base_name, str(task_id), str(reviser_size))
    os.makedirs(save_path, exist_ok=True)

    info_log_file_name = os.path.join(save_path, f"info_{timestamp}.log")
    debug_log_file_name = os.path.join(save_path, f"debug_{timestamp}.log")
    
    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    info_file_handler = logging.FileHandler(info_log_file_name)
    debug_file_handler = logging.FileHandler(debug_log_file_name)

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    info_file_handler.setLevel(logging.INFO)
    debug_file_handler.setLevel(logging.DEBUG)
    # logging.getLogger("openai").setLevel(logging.ERROR)
    # logging.getLogger("httpx").setLevel(logging.ERROR)

    # Create formatters and add them to the handlers
    console_formatter = logging.Formatter('%(message)s')
    info_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(console_formatter)
    info_file_handler.setFormatter(info_file_formatter)
    debug_file_handler.setFormatter(debug_file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(debug_file_handler)

    return logger
