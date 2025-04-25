import logging
from pathlib import Path

from config import LOG_PATH

logging.basicConfig(filename=Path(LOG_PATH), level=logging.DEBUG)
logger = logging.getLogger(__name__)
