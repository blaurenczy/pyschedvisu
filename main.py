#!/usr/bin/env python

"""
This is the master script running all the steps.
"""

import logging
import codecs
import os

import smtplib
from email.message import EmailMessage
from email.headerregistry import Address
from email.mime.application import MIMEApplication
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from configparser import ConfigParser

# remove MATPLOTLIBDATA warning
import warnings
warnings.filterwarnings("ignore")

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
    send_email(config, pdf_output_path)
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
    if not os.path.isfile(config_path):
        print(f'ERROR: Could not file config file at "{config_path}".')
        return None

    # read in the configuration file
    config = ConfigParser()
    config.optionxform = str
    with codecs.open(config_path, "r", "utf-8-sig") as f:
        config.read_file(f)

    return config

def get_day_range(config, reduce_freq=False):
    """
    Returns the starting day, the ending day and the days range specified by the config.
    Args:
        config (dict):              a dictionary holding all the necessary parameters
        reduce_freq (boolean):      whether or not to give reduced frequency
    Returns:
        start_date (datetime):  the starting day specified by the config
        end_date (datetime):    the starting day specified by the config
        days_range (daterange): the pandas daterange going from the starting to then ending day
    """

    # if auto mode for report
    if config['main']['end_date'] == 'auto' and config['main']['mode'] == 'report':
        # use the current day as starting point
        today = dt.today()
        # use previous day if it is not 22:00 yet
        if today.hour < 22: today -= timedelta(days=1)
        end_date = today - timedelta(days=today.weekday() - 4)
        if end_date > today: end_date = end_date - timedelta(days=7)
    else:
        end_date = dt.strptime(config['main']['end_date'], '%Y%m%d')

    # if auto mode for report
    if config['main']['start_date'] == 'auto' and config['main']['mode'] == 'report':
        start_date = dt(2016,1,1)
    else:
        start_date = dt.strptime(config['main']['start_date'], '%Y%m%d')

    days_range = pd.date_range(start_date, end_date, freq='B')

    # if a reduced frequency is required, recreate the range with a frequency depending on the number of days to show
    if reduce_freq:
        n_days = len(days_range)
        # for less than 2 weeks, show all days
        if n_days <= 10:
            days_range = pd.date_range(start_date, end_date, freq='B')
        # between 2 weeks and a month, group by week
        elif n_days <= 22:
            days_range = pd.date_range(start_date, end_date, freq='W-MON')
        # between a month and 4 months, group by 4 weeks periods
        elif n_days <= 86:
            days_range = pd.date_range(start_date, end_date, freq='4W-MON')
        # between 4 months and 12 months, group by month
        elif n_days <= 252:
            days_range = pd.date_range(start_date, end_date, freq='BMS')
        # more than 12 months, group by year
        else:
            days_range = pd.date_range(start_date, end_date, freq='BYS')

        logging.debug(f'Days range has {n_days} days, grouping by {days_range.freq.name}')

    return start_date, end_date, days_range

def send_email(config, pdf_output_path):
    """
    Send an email after creating a report in auto mode.
    Args:
        config (dict):          a dictionary holding all the necessary parameters
        pdf_output_path (str):  path to the PDF output file
    Returns:
        None
    """

    # skip email sending if specified by the config
    if not config['email'].getboolean('debug_send_email'): return

    # get the relevant part of the config
    email = config['email']

    # create the message
    msg = EmailMessage()

    # set the headers
    msg['Subject'] = email['subject']
    msg['From'] = Address(display_name=email['sender_name'], addr_spec=email['sender_email'])
    msg['To'] = [Address(addr_spec=email_addr) for email_addr in email['recipients_email'].split(',')]

    # get the email's body and replace the relevant parts
    body = email['body']
    pdf_file_name = pdf_output_path.split('/')[-1]
    body = body.replace('{__REPORT_PATH__}', pdf_output_path)
    body = body.replace('{__REPORT_FOLDER_PATH__}', '/'.join(pdf_output_path.split('/')[:-1]))
    body = body.replace('{__REPORT_FILE_NAME__}', pdf_file_name)

    # add the body as HTML
    msg.add_alternative(body, 'html')

    # read in the PDF as bytes
    with open(pdf_output_path, 'rb') as pdf:
        pdf_data = pdf.read()

    # add the PDF as attachment
    att = MIMEApplication(pdf_data, 'pdf')
    att.add_header('Content-Disposition', 'attachment', filename=pdf_file_name) # specify the filename
    msg.make_mixed() # This converts the message to multipart/mixed
    msg.attach(att)

    # create the connection to the server and send the mail
    with smtplib.SMTP(email['smtp_server']) as session:
        session.send_message(msg)

if __name__ == '__main__':
    run()
