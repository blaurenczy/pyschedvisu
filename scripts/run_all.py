#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
from retrieve_DICOMs_from_PACS import retrieve_DICOMs_from_PACS
from extract_data_from_DICOMs import extract_data_from_DICOMs
from create_report import create_report


def run_all():

    logging.basicConfig(format='%(asctime)s|%(levelname)s| %(message)s', level=logging.INFO)
    logging.info("Started running SchedVisu workflow")

    # get date range
    start_date = date(2018, 7, 2)
    end_date = date(2018, 7, 6)
    n_days = (end_date - start_date).days

    logging.info("Using dates: %s - %s (%d days)".format(start_date, end_date, n_days))

    retrieve_DICOMs_from_PACS()
    extract_data_from_DICOMs()
    create_report(start_date, end_date)

    logging.info("Finished running SchedVisu workflow.")

    return


if __name__ == '__main__':
    run_all()
