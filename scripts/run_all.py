#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
from datetime import date
from retrieve_DICOMs_from_PACS import retrieve_DICOMs_from_PACS
from extract_data_from_DICOMs import extract_data_from_DICOMs
from create_report import create_report


def run_all():

    logging.basicConfig(format='%(asctime)s|%(levelname)s| %(message)s', level=logging.INFO)
    logging.info("Started running SchedVisu workflow")

    # dictionary holding all the configuration information
    config = {}

    # create some test variables
    config['start_date'] = date(2018, 7, 2)
    config['end_date'] = date(2018, 7, 6)
    config['n_days'] = (config['end_date'] - config['start_date']).days + 1
    config['machine_name'] = 'Discovery 690'
    config['n_minutes_for_gap'] = 30
    config['n_months_average_for_slot_number'] = 6

    logging.info("Using dates: {start_date} - {end_date} ({n_days} days)".format(**config))
    logging.info("Using machine: {machine_name}".format(**config))

    retrieve_DICOMs_from_PACS()
    extract_data_from_DICOMs()
    create_report(config)

    logging.info("Finished running SchedVisu workflow.")

    return


if __name__ == '__main__':
    run_all()
