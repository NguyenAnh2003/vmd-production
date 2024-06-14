import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: -- %(message)s --: %(asctime)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.removeHandler(logging.FileHandler)  # remove file handler
    logger.setLevel(logging.DEBUG)
    return logger