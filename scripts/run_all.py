#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
from configparser import ConfigParser
from datetime import date
from datetime import datetime
from scripts.retrieve_data_from_PACS import retrieve_data_from_PACS
from scripts.create_report import create_report


def run_all():

    logging.basicConfig(format='%(asctime)s|%(levelname)s| %(message)s', level=logging.INFO)

    logging.info("Reading configuration")
    # read in the configuration file
    config = ConfigParser()
    config.optionxform = str
    config.read('config.ini')

    # create some test variables
    start_date = datetime.strptime(config['main']['start_date'], '%Y%m%d')
    end_date = datetime.strptime(config['main']['end_date'], '%Y%m%d')
    config.set('main', 'machine_name', 'Discovery 690')
    config.set('main', 'n_days', str((end_date - start_date).days + 1))

    # run the workflow
    logging.info("Starting SchedVisu workflow")
    # retrieve_data_from_PACS(config)
    # create_report(config)
    logging.info("Finished running SchedVisu workflow")

    return config


if __name__ == '__main__':
    run_all()
