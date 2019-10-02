#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

from retrieve_DICOMs_from_PACS import retrieve_DICOMs_from_PACS
from extract_data_from_DICOMs import extract_data_from_DICOMs
from create_report import create_report


def run_all():

    retrieve_DICOMs_from_PACS()
    extract_data_from_DICOMs()
    create_report()

    return


if __name__ == '__main__':
    run_all()
