#!/usr/bin/env python

import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.patches import FancyBboxPatch
from random import random
from datetime import date, timedelta
from datetime import datetime as dt
import scripts.main

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
    #for machine in ['PET Siemens']:

        # create a matplotlib figure with the right aspect ratio
        fig = plt.figure(figsize=[8.27, 11.69])

        # create the report, section by section
        create_header(config, fig, machine)
        #create_notes(config, fig)
        create_schedule(config, fig, machine, df)
        create_daily_table(config, fig, machine, df)
        #create_violin(c, config)
        #create_stat_table(c, config)

        logging.info("Saving PDF file")
        fig.savefig('output_{}.pdf'.format(machine.lower().replace(' ', '')), orientation='portrait',
            papertype='a4', format='pdf')

def create_header(config, fig, machine):
    """
    Create the header section with the logo, the header text, the dates, etc.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
    Returns:
        None
    """

    logging.info("Creating header section")

    # analyse the week numbers
    start_date, end_date, _ = scripts.main.get_day_range(config)
    week_numbers = sorted(list(set([start_date.strftime('%V'), end_date.strftime('%V')])))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(week_numbers)
    logging.info(f"Header content: {report_type}, {week_numbers_str}")

    # draw the header text with dates, etc.
    plt.rcParams["font.family"] = "monospace"
    fig.text(0.62, 0.97, "Rapport {}".format(report_type), fontsize=15)
    fig.text(0.62, 0.93, "Semaine{} {}".format('s' if len(week_numbers) > 1 else '', week_numbers_str), fontsize=25, fontweight='bold')
    fig.text(0.63, 0.89, "du {}".format(start_date.strftime("%d/%m/%Y")), fontsize=20)
    fig.text(0.63, 0.86, "au {}".format(end_date.strftime("%d/%m/%Y")), fontsize=20)

    # machine name
    fig.text(0.01, 0.83, 'Machine: ' + machine, fontsize=20)

    im_machine_path = '{}/images/{}.png'.format(os.getcwd(), machine.lower().replace(' ', '')).replace('/', '\\')
    im_machine = plt.imread(get_sample_data(im_machine_path))
    im_machine_ax = fig.add_axes([0.40, 0.81, 0.22, 0.05], anchor='NE')
    im_machine_ax.imshow(im_machine)
    im_machine_ax.axis('off')

    ## draw the logo
    im_logo_path = '{}/images/logo_transp.png'.format(os.getcwd()).replace('/', '\\')
    im_log = plt.imread(get_sample_data(im_logo_path))
    im_logo_ax = fig.add_axes([0.00, 0.86, 0.60, 0.13], anchor='NE')
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


def create_notes(config, fig):
    """
    Create the notes section.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
    Returns:
        None
    """

    logging.info("Creating notes section")

    # draw the notes
    #c.drawString(15, -600, "Note\u00b9: le nombre de 'slot' est calculé sur la base de la " +\
    #    "moyenne des {n_months_average_for_slot_number} derniers mois".format(**config))
    #c.drawString(15, -615, "Note\u00b2: les trous sont définis comme des espaces d'au " +\
    #    "moins {n_minutes_for_gap} minutes sans examens".format(**config))

def create_schedule(config, fig, machine, df):
    """
    Create the schedule section with the plot for the schedule and the distribution.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    logging.debug("Creating schedule section")

    logging.debug("Creating schedule plot")
    create_schedule_plot(config, fig, machine, df)

    logging.debug("Creating distribution plot")
    create_schedule_distribution_plot(config, fig, machine, df)

def create_schedule_plot(config, fig, machine, df):
    """
    Create the schedule plot for the schedule section.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = scripts.main.get_day_range(config)
    # remove the Sundays from the days range
    days_range = [d for d in days_range if d.weekday() != 6]

    # create the new axes
    sched_ax = fig.add_axes([0.06, 0.42, 0.80, 0.39], anchor='NE')

    # plot each day
    for day in days_range:
        plot_day_for_schedule_plot(config, sched_ax, machine, day, df)

    # create the x ticks
    days_xticks, days_xtick_labels = [], []
    for day in days_range:
        if day.weekday() in [5, 6]: continue
        days_xticks.append(days_range.index(day) + 1)
        days_xtick_labels.append(day.strftime('%d/%m'))

    # set the ticks, labels and the limits of the plot
    start_hour = config['draw'].getint('sched_start_hour')
    end_hour = config['draw'].getint('sched_end_hour')
    plt.xticks(days_xticks, days_xtick_labels)
    plt.yticks(
        ticks=range(start_hour, end_hour + 1),
        labels=['{:02d}h'.format(i) for i in range(start_hour, end_hour + 1)])
    plt.xlim((0.5, len(days_range) + 0.5))
    plt.ylim((start_hour - 0.5, end_hour + 0.5))

def plot_day_for_schedule_plot(config, sched_ax, machine, day, df):
    """
    Plot a single day in the schedule plot.
    Args:
        config (dict):      a dictionary holding all parameters for generating the report (dates, etc.)
        sched_ax (Axes):    the matplotlib axes object for drawing
        machine (str):      a string specifying the currently processed machine
        day (datetime):     the currently processed day
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """


    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = scripts.main.get_day_range(config)
    # remove the Sundays from the days range
    days_range = [d for d in days_range if d.weekday() != 6]
    # initialize some variables related to the dates
    day_str = day.strftime('%Y%m%d')
    i_day = days_range.index(day) + 1

    # get the data for the current day and machine
    df_day = df.query('Date == "{}" & Machine == "{}"'.format(day_str, machine))
    logging.debug('Found {} studies in day {} (day number: {})'.format(len(df_day), day_str, i_day))
    # abort if no data
    if len(df_day) == 0: return

    # go through each study found for this day
    for i_study in range(len(df_day)):
        study = df_day.iloc[i_study, :]

        # get the start time, end time and duration as hours with decimals
        start = pd.to_datetime(study['Start Time'], format='%H%M%S')
        end = pd.to_datetime(study['End Time'], format='%H%M%S')
        start_hour = start.hour + start.minute / 60 + start.second / 3600
        end_hour = end.hour + end.minute / 60 + end.second / 3600
        duration_hours = end_hour - start_hour

        # if the duration is negative
        if duration_hours <= 0:
            logging.warning('Problem with study {} on day {}: duration is 0 or negative: {}'
                .format(study.name, day_str, duration_hours))
            continue

        # check if we have an overlap issue
        if i_study > 0:
            end_prev = pd.to_datetime(df_day.iloc[i_study - 1, :]['End Time'], format='%H%M%S')
            end_prev_hour = end_prev.hour + end_prev.minute / 60 + end_prev.second / 3600
            if start_hour <= end_prev_hour:
                logging.warning(('Problem with study ...{} on {} & day {}: start hour {:6.3f} is ' +
                    'before end hour of last study {:6.3f}').format('.'.join(study.name.split('.')[-2:]),
                    machine, day_str, start_hour, end_prev_hour))

        # get the start and stop times rounded to the minute
        logging.debug('day {}, start {:5.2f} -> end {:5.2f}, duration: {:4.2f}'
            .format(day_str, start_hour, end_hour, duration_hours))

        # get the coordinates where the rounded rectangle for this study should be plotted
        box_w = config['draw'].getfloat('study_box_w')
        x_shift = config['draw'].getfloat('study_x_shift')
        x = i_day - (box_w * 0.5) + (x_shift * (-1 if (i_study % 2 == 0) else 1))
        y, w, h = start_hour, box_w, duration_hours

        # create the shape and plot it
        rounded_rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=-0.0040,rounding_size=0.155",
            fc="red", ec='black', mutation_aspect=0.4)
        sched_ax.add_patch(rounded_rect)

        # DEBUG show information string
        plt.text(x + w * 0.1, y + 0.1 * h, '{}: {} - {}: {:.2f}h'
            .format(*study[['Patient ID', 'Start Time', 'End Time']], duration_hours))

def create_schedule_distribution_plot(config, fig, machine, df):
    """
    Create the schedule's distribution plot for the schedule section.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    # define some parameters for the distribution
    FMT = '%H%M%S'
    start_date, end_date, _ = scripts.main.get_day_range(config)
    start_hour = '{:02d}3000'.format(config['draw'].getint('sched_start_hour') - 1)
    end_hour = '{:02d}3000'.format(config['draw'].getint('sched_end_hour'))
    # create a list of times which will be used as time points to build the distribution
    time_range = pd.date_range(dt.strptime(start_hour, FMT), dt.strptime(end_hour, FMT),
        freq=config['draw']['sched_distr_freq'])

    logging.debug("Calculating distribution")
    # get all the studies for the selected time period
    df_period = df.query('Date >= "{}" & Date <= "{}" & Machine == "{}"'
        .format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), machine))
    # initialize the DataFrame storing the counts for each time point
    df_counts = pd.DataFrame([0]*len(time_range), index=time_range, columns=['count'])

    # loop through each study
    for ind in df_period.index:
        # get the start and end time of each study
        start = dt.strptime(df_period.loc[ind, 'Start Time'], FMT)
        end = dt.strptime(df_period.loc[ind, 'End Time'], FMT)
        # go through each time point
        for t in time_range:
            # check where the time point is within the study's time range. If yes, increment the counter
            if start <= t <= end: df_counts.loc[t, 'count'] += 1
    # resample the distribution with a lower frequency to smooth the curve
    df_counts_resample = df_counts.resample(config['draw']['sched_distr_resample_freq']).mean()

    # plot the distribution
    logging.debug("Plotting the distribution")
    # add new axes
    distr_ax = fig.add_axes([0.86, 0.42, 0.12, 0.39], anchor='NE')
    # create new y values based on the time points
    counts_y_values = [t.hour + t.minute / 60 + t.second / 3600 for t in df_counts_resample.index]
    # plot the curve
    plt.plot(df_counts_resample['count'], counts_y_values, color='red')
    # set some limits and remove the ticks
    plt.ylim((7.5, 19.5))
    plt.xticks([])
    plt.yticks([])

def create_daily_table(config, fig, machine, df):
    """
    Create the daily table section with the statistics for each day.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    logging.info("Creating daily table section")

    # add new axes
    table_ax = fig.add_axes([0.06, 0.31, 0.80, 0.12], anchor='NE')
    table_ax.axis('off')

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = scripts.main.get_day_range(config)
    # remove the Sundays from the days range
    days_range = [d for d in days_range if d.weekday() != 6]
    # initialize the variable holding all the information to be displayed in the table
    data = [[],[],[]]
    # go through each day
    for day in days_range:
        # get the data and calculate values for the current day
        df_day = df.query('Date == "{}" & Machine == "{}"'.format(day.strftime('%Y%m%d'), machine))
        n_max_studies = config['draw'].getint('n_study_per_day_' + machine.lower().replace(' ', ''))
        n_studies = len(df_day)
        # skip the weekends
        if day.weekday() in [5, 6]:
            [data[i].append('') for i in range(3)]
            continue
        # insert the values into the data table
        data[0].append(n_max_studies)
        data[1].append(n_studies)
        data[2].append('{:3d}%'.format(int(100 * n_studies / n_max_studies)))

    table = plt.table(cellText=data, cellLoc='center', loc='center')
    table.set_fontsize(9)
    table.auto_set_font_size(False)

    # add new axes for the row headers
    table_header_ax = fig.add_axes([0.01, 0.31, 0.05, 0.12], anchor='NE')
    table_header_ax.axis('off')
    table_header = plt.table(cellText=[['Slot'], ['Exam.'], ['Util.']], cellLoc='center', loc='center')
    table_header.set_fontsize(9)
    table_header.auto_set_font_size(False)

    # calculate summary values
    n_cols = len(data[0])
    tot_slots = sum([data[0][i_col] for i_col in range(n_cols) if isinstance(data[0][i_col], int)])
    tot_n_studies = sum([data[1][i_col] for i_col in range(n_cols) if isinstance(data[1][i_col], int)])
    summ_data = [
        [tot_slots, '{:.1f}'.format(tot_slots / n_cols)],
        [tot_n_studies, '{:.1f}'.format(tot_n_studies / n_cols)],
        ['', '{:.1f}'.format(100 * tot_n_studies / tot_slots)]]

    # add new axes for the summary values
    table_summ_ax = fig.add_axes([0.86, 0.31, 0.12, 0.12], anchor='NE')
    table_summ_ax.axis('off')
    table_summ = plt.table(cellText=summ_data, cellLoc='center', loc='center')
    table_summ.set_fontsize(10)
    table_summ.auto_set_font_size(False)

    # add new axes for the summary values
    table_summ_header_ax = fig.add_axes([0.86, 0.389, 0.12, 0.02], anchor='NE')
    table_summ_header_ax.axis('off')
    table_summ_header = plt.table(cellText=[['Total', 'Moyen']], cellLoc='center', loc='center')
    table_summ_header.set_fontsize(11)
    table_summ_header.auto_set_font_size(False)


def create_violin(c, config):
    """
    Create the violint plot section with the data for each examination type.
    Args:
        c (Canvas):     the canvas object for drawing
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
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
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
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
