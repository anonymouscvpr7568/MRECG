import os
import sys
import logging
import functools
from termcolor import colored

MQBENCH_LOGGER_NAME = "MQBENCH"
logger = logging.getLogger(MQBENCH_LOGGER_NAME)
logger.propagate = False
stdout_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)
logger.parent = None


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name='"MQBENCH"'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
def set_log_level(logger, level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging(logger):
    logger.handlers = []