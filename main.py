#!/usr/bin/env python

# remove MATPLOTLIBDATA warning
import warnings
warnings.filterwarnings("ignore")

import logging
import codecs
import os

from datetime import datetime as dt
from configparser import ConfigParser

import utils
import retrieve_data
import extract_data
import create_report

def run():
    """
    Main function to run the whole pipeline. Includes starting the logging, reading the config and
        running the entire pipeline with error catching.
    Args:
        None
    Returns:
        None
    """

    # load the configuration
    config = load_config()

    # create the logger
    create_logger(config)

    try:
        # run the workflow using the date range, settings, parameters, etc. found in the config
        run_pipeline(config)

    except Exception as e:
        logging.error('Error while running workflow')
        logging.error("-"*60)
        logging.error(e, exc_info=True)
        logging.error("-"*60)

    except KeyboardInterrupt:
        logging.error('Interrupted by user')

    finally:
        logging.shutdown()

def run_pipeline(config):
    """
    Function running the entire pipeline with the specified config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """
    logging.warning("Starting pySchedVisu workflow for range {start_date} - {end_date}".format(**config['main']))
    retrieve_data.retrieve_and_save_data_from_PACS(config)
    extract_data.load_transform_and_save_data_from_files(config)
    pdf_output_path = create_report.create_report(config)
    utils.send_email(config, pdf_output_path)
    logging.warning("Finished running pySchedVisu workflow for range {start_date} - {end_date}".format(**config['main']))

def create_logger(config):
    """
    Create a logger.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    # reset the logger
    root = logging.getLogger().handlers = []
    # define the logging format and paths
    logging_format = '%(asctime)s|%(funcName)-30.30s:%(lineno)03s|%(levelname)-7s| %(message)s'
    logging_dir = os.path.join(config['path']['log_dir'], '{}'.format(dt.now().strftime('%Y%m')))
    logging_filename = os.path.join(logging_dir, '{}_pySchedVisu.log'.format(dt.now().strftime('%Y%m%d_%H%M%S')))
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

    # set debug level based on the configuration file's content
    logging.getLogger().setLevel(config['main']['debug_level'].upper())
    # set debug level of pynetdicom based on the configuration file's content
    logging.getLogger('pynetdicom').setLevel(config['main']['pynetdicom_debug_level'].upper())

def load_config():
    """
    Retrieve and save the relevant series from the PACS for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    config_path = 'config.ini'
    # if the config file is not in the current directory
    if not os.path.isfile(config_path):
        config_path = '../../config.ini'
    # if the config file cannot be found
    if not os.path.isfile(config_path):
        print('ERROR: Could not find config file ("config.ini").')
        print('ERROR: Are you sure you copied the config file to the same directory where "main.exe" is?')
        input()
        return None

    # read in the configuration file
    config = ConfigParser()
    config.optionxform = str
    with codecs.open(config_path, "r", "utf-8-sig") as f:
        config.read_file(f)

    return config

if __name__ == '__main__':
    run()
