#!/usr/bin/env python

import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from svglib.svglib import svg2rlg
from reportlab.lib.pagesizes import A4


def create_report(config):
    """
    Generates a PDF report by extracting the relevant data from the database, processing it and drawing the relevant
    plots, tables and annotations.
    Args:
        config (dict): a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Reading in database")
    df = pd.read_csv('database.csv')

    logging.info("Initializing the canvas of the PDF file")
    c = canvas.Canvas('output/schedvisu.pdf', pagesize=A4)
    cw, ch = A4

    # Translate origin to upper left corner, to make it easier to use the coordinates from the GIMP draft.
    #   However, all Y coordinates will need to be negative.
    c.translate(0, ch)

    # # DEBUG
    # logging.info("Drawing report background")
    # c.drawImage('images/RapportHebdomadaire.png', 0, -ch, cw, ch)

    # create the report, section by section
    create_header(c, config)
    create_notes(c, config)
    create_schedule(c, config)
    create_daily_table(c, config)
    create_violin(c, config)
    create_stat_table(c, config)

    logging.info("Saving PDF file")
    c.showPage()
    c.save()

    return

def create_header(c, config):
    """
    Create the header section with the logo, the header text, the dates, etc.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating header section")

    # analyse the week numbers
    week_numbers = list(set([config['start_date'].strftime('%V'), config['end_date'].strftime('%V')]))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(week_numbers)
    logging.info(f"Header content: {report_type}, {week_numbers_str}")

    # draw the header text with dates, etc.
    c.setStrokeColorRGB(0, 0, 0)
    c.setFont("Times-Roman", 20)
    c.drawString(370, -25, "Rapport {}".format(report_type))
    c.setFont("Times-Bold", 30)
    c.drawString(370, -55, "Semaine {}".format(week_numbers_str))
    c.setFont("Times-Roman", 25)
    c.drawString(375, -85, "du {}".format(config['start_date'].strftime("%d/%m/%Y")))
    c.drawString(375, -115, "au {}".format(config['end_date'].strftime("%d/%m/%Y")))

    # machine name
    c.setFont("Times-Roman", 30)
    c.drawString(185, -145, config['machine_name'])
    c.drawImage('images/{}.png'.format(config['machine_name'].lower().replace(' ', '')), 65, -160, 115, 55)

    # draw the logo
    c.drawImage('images/logo_transp.png', 10, -110, 345, 105, mask='auto')


def get_report_type(week_numbers):
    """
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


def create_notes(c, config):
    """
    Create the notes section.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating notes section")

        # draw the notes
    c.setStrokeColorRGB(0, 0, 0)
    c.setFont("Times-Italic", 15)
    c.drawString(15, -600, "Note\u00b9: le nombre de 'slot' est calculé sur la base de la " +\
        "moyenne des {n_months_average_for_slot_number} derniers mois".format(**config))
    c.drawString(15, -615, "Note\u00b2: les trous sont définis comme des espaces d'au " +\
        "moins {n_minutes_for_gap} minutes sans examens".format(**config))


def create_schedule(c, config):
    """
    Create the schedule section with the plot for the schedule and the distribution.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating schedule section")

    logging.info("Creating schedule plot")
    # create the matplotlib figure with the right aspect ratio
    fig = plt.figure(figsize=(4.9, 3.5))
    plt.plot([1,2,3,4])
    plt.ylabel('Some Numbers')

    logging.info("Plotting schedule plot to canvas")
    draw_plot_to_canvas(c, fig, [10, 510, 490, 350], 'red')

    logging.info("Creating distribution plot")
    # create the matplotlib figure with the right aspect ratio
    fig = plt.figure(figsize=(0.8, 3.25))

    logging.info("Plotting distribution plot to canvas")
    draw_plot_to_canvas(c, fig, [510, 485, 80, 325], 'blue')


def draw_plot_to_canvas(c, fig, pos, face_color=None):
    """
    Create the schedule section with the plot for the schedule and the distribution.
    Args:
        c (Canvas):         the canvas object for drawing
        fig (Figure):       the matplotlib figure to be drawn
        pos (list of int):  a list of [x, y, w, h] coorinates where to draw
        face_color (str):   (optional) a string representing the background color for the figure
    Returns:
        None
    """

    # save the matplotlib figure as SVG to BytesIO object
    img_data = BytesIO()
    fig.savefig(img_data, format='svg', facecolor=face_color)
    # rewind the bytes array and convert to a reportlab Drawing object
    img_data.seek(0)
    drawing = svg2rlg(img_data)
    # rescale the drawing to make sure it has the right pixel size
    drawing.scale(pos[2] / drawing.width, pos[3] / drawing.height)
    # draw the object to the canvas
    renderPDF.draw(drawing, c, pos[0], -pos[1])


def create_daily_table(c, config):
    """
    Create the daily table section with the statistics for each day.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating daily table section")

    data = [
        ['19', '19', '19', '19', '19'],
        ['18', '19', '17', '20', '19'],
        ['95%', '100%', '89%', '105%', '100%'],
        ['00', '01', '02', '00', '00']
    ]

    t = Table(data, colWidths=[97] * len(data[0]))

    table_style = TableStyle([
        ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,-1),1,colors.black),
        ('BOX',(0,0),(-1,-1),1,colors.black),
        ('BOX',(0,0),(0,-1),1,colors.black)
    ])
    table_style.add('BACKGROUND',(0,0),(1,0),colors.lightblue)
    table_style.add('BACKGROUND',(0,1),(-1,-1),colors.white)
    t.setStyle(table_style)


    t.wrapOn(c, 485, 65)
    t.drawOn(c, 15, -585)


def create_violin(c, config):
    """
    Create the violint plot section with the data for each examination type.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating violin section")

    logging.info("Creating violin plot")
    # create the matplotlib figure with the right aspect ratio
    fig = plt.figure(figsize=(3, 2))
    plt.plot([1,2,3,4])
    plt.ylabel('Some Numbers')

    logging.info("Plotting violin plot to canvas")
    draw_plot_to_canvas(c, fig, [10, 830, 300, 200], 'green')


def create_stat_table(c, config):
    """
    Create the statistics table section with the statistics for each examination type.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
    Returns:
        None
    """

    logging.info("Creating statistics table section")

    data = [
        ['Nom', 'N. Exa', 'Dur. moy.', 'Dur. min', 'Dur.max'],
        ['FDG Cerveau', '19', '17', '20', '19'],
        ['FDG ORL', '100%', '89%', '105%', '100%'],
        ['Rb82 Coeur', '01', '02', '00', '00']
    ]

    t = Table(data, colWidths=[52] * len(data[0]), rowHeights=[47.25] * len(data))

    table_style = TableStyle([
        ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,-1),1,colors.black),
        ('BOX',(0,0),(-1,-1),1,colors.black),
        ('BOX',(0,0),(0,-1),1,colors.black)
    ])
    table_style.add('BACKGROUND',(0,0),(1,0),colors.lightblue)
    table_style.add('BACKGROUND',(0,1),(-1,-1),colors.white)
    t.setStyle(table_style)


    t.wrapOn(c, 270, 190)
    t.drawOn(c, 325, -825)
