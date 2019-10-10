#!/usr/bin/env python

import logging


def extract_data_from_DICOMs(config):
    """
    Extract the relevant information from the DICOM files and store them in a database.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    logging.info("Extracting data from DICOMs")

    # folder structure: {ROOT}/PACS_data/{YYYY_mm}/{YYYYmmdd}/{DICOM_FILE}

    return


if __name__ == '__main__':
    extract_data_from_DICOMs()
