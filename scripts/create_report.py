#!/usr/bin/env python

import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from datetime import date, timedelta
from datetime import datetime as dt



def create_report(config):
    """
    Generates a PDF report by extracting the relevant data from the database, processing it and drawing the relevant
    plots, tables and annotations.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    # create the path where the data for the current config would be stored
    day_period_str = '{}_{}'.format(config['main']['start_date'], config['main']['end_date']).replace('-', '')
    studies_save_path = 'data/studies/studies_{}.pkl'.format(day_period_str)

    # check if the data can be loaded
    if not os.path.isfile(studies_save_path):
        logging.error('Reading {}: Could not find save file at "{}", aborting.'
            .format(day_period_str, studies_save_path))
        return

    logging.info("Reading in studies")
    df = pd.read_pickle(studies_save_path)

    # exclude some machines and do some grouping up
    df['Machine'] = df['Machine Group'].str.replace('NoCT', '')
    df = df[df['Machine'] != 'mixed cases']

    # go through each machine
    for machine in set(df['Machine']):

        # create a matplotlib figure with the right aspect ratio
        fig = plt.figure(figsize=[8.27, 11.69])

        # create the report, section by section
        create_header(config, fig, machine)
        #create_notes(c, config)
        #create_schedule(c, config)
        #create_daily_table(c, config)
        #create_violin(c, config)
        #create_stat_table(c, config)

        logging.info("Saving PDF file")
        fig.savefig('output_{}.pdf'.format(machine.lower().replace(' ', '')), orientation='portrait',
            papertype='a4', format='pdf')

def create_header(config, fig, machine):
    """
    Create the header section with the logo, the header text, the dates, etc.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, machine name, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
    Returns:
        None
    """

    logging.info("Creating header section")

    start_day = dt.strptime(config['main']['start_date'], '%Y-%m-%d')
    end_day = dt.strptime(config['main']['end_date'], '%Y-%m-%d')

    # analyse the week numbers
    week_numbers = list(set([start_day.strftime('%V'), end_day.strftime('%V')]))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(week_numbers)
    logging.info(f"Header content: {report_type}, {week_numbers_str}")

    # draw the header text with dates, etc.
    plt.rcParams["font.family"] = "monospace"
    fig.text(0.62, 0.97, "Rapport {}".format(report_type), fontsize=15)
    fig.text(0.62, 0.93, "Semaine{} {}".format('s' if len(week_numbers) > 1 else '', week_numbers_str), fontsize=25, fontweight='bold')
    fig.text(0.63, 0.89, "du {}".format(start_day.strftime("%d/%m/%Y")), fontsize=20)
    fig.text(0.63, 0.86, "au {}".format(end_day.strftime("%d/%m/%Y")), fontsize=20)

    # machine name
    fig.text(0.01, 0.82, 'Machine: ' + machine, fontsize=20)

    im_machine_path = '{}/images/{}.png'.format(os.getcwd(), machine.lower().replace(' ', '')).replace('/', '\\')
    im_machine = plt.imread(get_sample_data(im_machine_path))
    im_machine_ax = fig.add_axes([0.40, 0.81, 0.22, 0.05], anchor='NE', zorder=-1)
    im_machine_ax.imshow(im_machine)
    im_machine_ax.axis('off')

    ## draw the logo
    im_logo_path = '{}/images/logo_transp.png'.format(os.getcwd()).replace('/', '\\')
    im_log = plt.imread(get_sample_data(im_logo_path))
    im_logo_ax = fig.add_axes([0.00, 0.86, 0.60, 0.13], anchor='NE', zorder=-1)
    im_logo_ax.imshow(im_log)
    im_logo_ax.axis('off')


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
