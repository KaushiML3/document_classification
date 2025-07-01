import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Get the current working directory
#current_direction = os.path.dirname(os.path.abspath(__file__))
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

# Define the logging configuration
def setup_logger(file_name:str=None,api_app=None):

    if file_name is not None :
        LOG_FILE_PATH=os.path.join(logs_path,f"{file_name}.log")
        #log_formatter = logging.Formatter("%(asctime)s- %(name)s - %(levelname)s - %(message)s")

        # Modified log formatter to include filename, function name, and line number
        log_formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(levelname)s - %(message)s")

        # File handler for logging to a file
        file_handler = RotatingFileHandler(filename=LOG_FILE_PATH,maxBytes=5 * 1024 * 1024, backupCount=3)  # Log file size is 5MB with 3 backups
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)


        file_handler2 = RotatingFileHandler(filename=os.path.join(logs_path,"global.log"),maxBytes=5 * 1024 * 1024, backupCount=3)  # Log file size is 5MB with 3 backups
        file_handler2.setFormatter(log_formatter)
        file_handler2.setLevel(logging.INFO)

        # Stream handler for console output (optional)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.DEBUG)


        # Add handlers to the root logger for custom logging
        root_logger = logging.getLogger(file_name)
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(file_handler2)
        #root_logger.addHandler(console_handler)

        if api_app is not None:
            # Get the FastAPI logger and attach handlers
            uvicorn_access_logger = logging.getLogger("uvicorn.access")  # For request logging
            uvicorn_access_logger.setLevel(logging.INFO)
            uvicorn_access_logger.addHandler(file_handler)
            uvicorn_access_logger.addHandler(file_handler2)
            #api_logger.addHandler(console_handler)

            return uvicorn_access_logger
    
        else:
            return root_logger


    else:

        # Modified log formatter to include filename, function name, and line number
        log_formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(levelname)s - %(message)s")


        file_handler2 = RotatingFileHandler(filename=os.path.join(logs_path,"global.log"),maxBytes=5 * 1024 * 1024, backupCount=3)  # Log file size is 5MB with 3 backups
        file_handler2.setFormatter(log_formatter)
        file_handler2.setLevel(logging.INFO)

        # Stream handler for console output (optional)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.DEBUG)


        # Add handlers to the root logger for custom logging
        root_logger = logging.getLogger(file_name)
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler2)
        #root_logger.addHandler(console_handler)

        if api_app is not None:
            # Get the FastAPI logger and attach handlers
            uvicorn_access_logger = logging.getLogger("uvicorn.access")  # For request logging
            uvicorn_access_logger.setLevel(logging.INFO)
            uvicorn_access_logger.addHandler(file_handler2)
            #api_logger.addHandler(console_handler)

            return uvicorn_access_logger
        
        else:
            return root_logger