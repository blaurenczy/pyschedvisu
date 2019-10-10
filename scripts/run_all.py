#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
from configparser import ConfigParser
from datetime import date
from retrieve_DICOMs_from_PACS import retrieve_DICOMs_from_PACS
from extract_data_from_DICOMs import extract_data_from_DICOMs
from create_report import create_report


def run_all():

    logging.basicConfig(format='%(asctime)s|%(levelname)s| %(message)s', level=logging.INFO)

    logging.info("Reading configuration")
    # read in the configuration file
	config = ConfigParser()
	config.read('config.ini')

    # create some test variables
    config['start_date'] = date(2018, 7, 2)
    config['end_date'] = date(2018, 7, 6)
    config['n_days'] = (config['end_date'] - config['start_date']).days + 1
    config['machine_name'] = 'Discovery 690'

	# run the workflow
    logging.info("Starting SchedVisu workflow")
    retrieve_DICOMs_from_PACS(config)
    # extract_data_from_DICOMs(config)
    # create_report(config)
    logging.info("Finished running SchedVisu workflow")

    return


if __name__ == '__main__':
    run_all()
