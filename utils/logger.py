import logging
from constants import LOCAL_LOG_PATH

formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")


logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler(LOCAL_LOG_PATH), logging.StreamHandler()],
)

logger = logging.getLogger("create-segmentation")
