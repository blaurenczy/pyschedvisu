#!/usr/bin/env python

"""
This is the script creating the report.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from reportlab.lib.pagesizes import A4


def create_report(machine_name, start_date, end_date):
    """
    Create a PDF report for a specific machine with a given set of dates.

    Generates a PDF report by extracting the relevant data from the database, processing it and drawing the relevant
    plots, tables and annotations.

    Args:
        machine_name (string):  The name of the machine for which to report has to be generated
        start_date (date):      The starting date for the report as a datetime.date object
        end_date (date):        The ending date for the report as a datetime.date object

    Returns:
        None

    """

    logging.info("Reading in database")
    df = pd.read_csv('database.csv')

    logging.info("Creating PDF file")
    c = canvas.Canvas('output/schedvisu.pdf', pagesize=A4)
    cw, ch = A4

    logging.info("Drawing report background")
    c.drawImage('images/RapportHebdomadaire.png', 0, 0, cw, ch)

    header(c, ch, machine_name, start_date, end_date);
    logging.info(type(ch))

    # logging.info("Creating schedule plot")
    # fig = plt.figure(figsize=(4, 3))
    # plt.plot([1,2,3,4])
    # plt.ylabel('Some Numbers')
    #
    # logging.info("Saving schedule plot to bytes")
    # img_data = BytesIO()
    # fig.savefig(img_data, format='svg')
    # img_data.seek(0)
    #
    # logging.info("Converting schedule plot to SVG drawing")
    # drawing = svg2rlg(img_data)
    # renderPDF.draw(drawing, c, 10, 40)
    # c.drawString(10, 300, "So nice it works")

    logging.info("Saving PDF file")
    c.showPage()
    c.save()

    return

def header(c, ch, machine_name, start_date, end_date):
    """
    Create the header section.

    Create the header section with the logo, the header text, the dates, etc.

    Args:
        c (Canvas):             The canvas object for drawing
        ch (float):             The height of the page, for adjusting the positions
        machine_name (string):  The name of the machine for which to report has to be generated
        start_date (date):      The starting date for the report as a datetime.date object
        end_date (date):        The ending date for the report as a datetime.date object

    Returns:
        None

    """

    logging.info("Adding header")

    # analyse the week numbers
    week_numbers = list(set([start_date.strftime('%V'), end_date.strftime('%V')]))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(week_numbers)
    logging.info(f"Header content: {report_type}, {week_numbers_str}")

    # draw the logo
    c.drawImage('images/logo.png', 10, ch - 115, 350, 110)

    # draw the header text with dates, etc.
    c.setStrokeColorRGB(0, 0, 0)
    c.setFont("Times-Roman", 20)
    c.drawString(370, ch - 25, "Rapport {}".format(report_type))
    c.setFont("Times-Bold", 30)
    c.drawString(370, ch - 55, "Semaine {}".format(week_numbers_str))
    c.setFont("Times-Roman", 25)
    c.drawString(375, ch - 85, "du {}".format(start_date.strftime("%d/%m/%Y")))
    c.drawString(375, ch - 115, "au {}".format(end_date.strftime("%d/%m/%Y")))

    # machine name
    c.setFont("Times-Roman", 30)
    c.drawString(10, ch - 145, "Machine: {}".format(machine_name))


def get_report_type(week_numbers):
    """
    Return the report type based on the week numbers.

    Return the report type ('hebdomadaire', 'bimensuel', 'mensuel', etc.) based on the week numbers provided.

    Args:
        week_numbers (list of string): A two-element list of strings representing week numbers

    Returns:
        report_type (string): a string representing the report type ('hebdomadaire', 'bimensuel', 'mensuel', etc.)

    """

    if len(week_numbers) == 1:
        report_type = 'hebdomadaire'
    else:
        week_diff = int(week_numbers[1]) - int(week_numbers[0])
        if week_diff == 1:                  report_type = 'bimensuel'
        elif week_diff == 3:                report_type = 'mensuel'
        elif week_diff in (51, 52, 53):     report_type = 'annuel'
        else:                               report_type = 'de {} semaines'.format(week_diff + 1)

    return report_type
