import logging
import os

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Configure logging

logging.basicConfig(
    filename="logs/system.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


def log_event(message):

    logging.info(message)
