# remove MATPLOTLIBDATA warning
import warnings
warnings.filterwarnings("ignore")

import logging

import smtplib
from email.message import EmailMessage
from email.headerregistry import Address
from email.mime.application import MIMEApplication

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta


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
        start_date = dt(config['main'].getint('report_year_start'), 1, 1)
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
