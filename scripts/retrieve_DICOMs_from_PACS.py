#!/usr/bin/env python

import logging


def retrieve_DICOMs_from_PACS(config):
    """
    Retrieve the relevant DICOM files from the PACS and download them in a temporary folder.
    Args:
        config (dict): a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Retrieving DICOMs from PACS")

    return


if __name__ == '__main__':
    retrieve_DICOMs_from_PACS()
