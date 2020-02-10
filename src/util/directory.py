import re
import os
import logging
from datetime import datetime


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def folder_name_generator(cf):
    name_str = []
    name_str.append('{}'.format(cf.model_name))
    
    name_str.append(get_time())
    return '-'.join(name_str)


def prepare_dirs_and_logger(cf):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    
    if not os.path.exists('./logger'):
        os.makedirs('./logger')
    if not os.path.exists('./data'):
        os.makedirs('./data')    
    cf.dir = folder_name_generator(cf)
    cf.dir = './logger/{}'.format(cf.dir)
    if not os.path.exists(cf.dir):
        os.makedirs(cf.dir)