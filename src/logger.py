import logging
import os
from datetime import datetime

# Generate a unique log file name based on the current datetime
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the path to the logs directory
logs_directory = os.path.join(os.getcwd(), "logs")

# Ensure the logs directory exists
try:
    os.makedirs(logs_directory, exist_ok=True)
except Exception as e:
    print(f"Failed to create logs directory: {e}")
    raise

# Create the full path for the log file
LOG_FILE_PATH = os.path.join(logs_directory, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Set the log message format
    level=logging.INFO,  # Set the default logging level to INFO
)

# Check logging is working:
if __name__=='__main__':
    logging.info("Logging is working")