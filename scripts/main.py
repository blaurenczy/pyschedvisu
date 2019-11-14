#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
from configparser import ConfigParser

from scripts.retrieve_data import retrieve_and_save_data_from_PACS
from scripts.extract_data import extract_transform_and_save_data_from_files
from scripts.create_report import create_report


def run():

    # create the logger
    logging.basicConfig(format='%(asctime)s|%(funcName)-30.30s:%(lineno)03s|%(levelname)-7s| %(message)s', level=logging.INFO)

    logging.info("Reading configuration")
    # read in the configuration file
    config = ConfigParser()
    config.optionxform = str
    config.read('config.ini')
    
    # set debug level based on the configuration file's content
    logging.getLogger().setLevel(config['main']['debug_level'])

    # run the workflow using the date range, settings, parameters, etc. found in the config
    logging.info("Starting SchedVisu workflow")
    retrieve_and_save_data_from_PACS(config)
    extract_transform_and_save_data_from_files(config)
    # create_report(config)
    logging.info("Finished running SchedVisu workflow")


if __name__ == '__main__':
    run()