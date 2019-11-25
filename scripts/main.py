#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
import os
from datetime import datetime as dt
from configparser import ConfigParser

from scripts.retrieve_data import retrieve_and_save_data_from_PACS
from scripts.extract_data import extract_transform_and_save_data_from_files
from scripts.create_report import create_report

def run():
    """
    Main function running the entire pipeline.
    Args:
        None
    Returns:
        None
    """

    # create the logger
    create_logger()

    # load the configuration
    logging.info("Reading configuration")
    config = load_config()

    # run the workflow using the date range, settings, parameters, etc. found in the config
    logging.info("Starting SchedVisu workflow")
    retrieve_and_save_data_from_PACS(config)
    # extract_transform_and_save_data_from_files(config)
    create_report(config)
    logging.info("Finished running SchedVisu workflow")

def create_logger():
    """
    Create a logger.
    Args:
        None
    Returns:
        None
    """

    # reset the logger
    root = logging.getLogger().handlers = []
    # define the logging format and paths
    logging_format = '%(asctime)s|%(funcName)-30.30s:%(lineno)03s|%(levelname)-7s| %(message)s'
    logging_dir = os.path.join('logs', '{}'.format(dt.now().strftime('%Y%m')))
    logging_filename = os.path.join(logging_dir, '{}_schedvisu.log'.format(dt.now().strftime('%Y%m%d_%H%M%S')))
    # make sure the log directory exists
    if not os.path.exists(logging_dir): os.makedirs(logging_dir)
    # create the logger writing to the file
    logging.basicConfig(format=logging_format, level=logging.INFO, filename=logging_filename)
    # define a Handler which writes messages to the console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # tell the handler to use this format
    console.setFormatter(logging.Formatter(logging_format))
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def load_config():
    """
    Retrieve and save the relevant series from the PACS for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    # read in the configuration file
    config = ConfigParser()
    config.optionxform = str
    config.read('config.ini')

    # set debug level based on the configuration file's content
    logging.getLogger().setLevel(config['main']['debug_level'].upper())
    # set debug level of pynetdicom based on the configuration file's content
    logging.getLogger('pynetdicom').setLevel(config['main']['pynetdicom_debug_level'].upper())

    return config

if __name__ == '__main__':
    run()
