#!/usr/bin/env python

import logging
import pandas as pd
import os
import math
import shutil

import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.colors as mc
import colorsys

from collections import namedtuple
from copy import deepcopy
from random import random
from datetime import date, timedelta
from datetime import datetime as dt

from PyPDF2 import PdfFileWriter, PdfFileReader

import main
import extract_data

def create_report(config):
    """
    Generates a PDF report by extracting the relevant data from the database, processing it and drawing the relevant
    plots, tables and annotations.
    Args:
        config (dict):          a dictionary holding all the necessary parameters
    Returns:
        pdf_output_path (str):  the path where the report is stored
    """

    # get the date ranges
    start_date_global, end_date_global, _ = main.get_day_range(config)

    # output a single page with all the data in one report per machine
    if config['main']['mode'] == 'single':
        date_ranges = [{'start': start_date_global, 'end': end_date_global}]

    # output a multiples pages with the data in multiple report per machine and per date ranges
    if config['main']['mode'] == 'report':

        prev_friday = end_date_global - timedelta(days=end_date_global.weekday() - 4)
        if prev_friday > end_date_global: prev_friday = prev_friday - timedelta(days=7)
        previous_monday_1W = prev_friday - timedelta(days= 5 +  0 * 7 - 1)
        previous_monday_2W = prev_friday - timedelta(days= 5 +  1 * 7 - 1)
        previous_monday_4W = prev_friday - timedelta(days= 5 +  3 * 7 - 1)
        previous_monday_3M = prev_friday - timedelta(days= 5 + 11 * 7 - 1)
        previous_monday_6M = prev_friday - timedelta(days= 5 + 23 * 7 - 1)
        previous_monday_1Y = prev_friday.replace(day=1).replace(month=1)
        previous_monday_4Y = previous_monday_1Y.replace(year=prev_friday.year - 3)

        date_ranges = []
        if 'hebdomadaire'   in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_1W, 'end': prev_friday })
        if 'bimensuel'      in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_2W, 'end': prev_friday })
        if 'mensuel'        in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_4W, 'end': prev_friday })
        if 'trimestriel'    in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_3M, 'end': prev_friday })
        if 'semestriel'     in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_6M, 'end': prev_friday })
        if 'annuel'         in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_1Y, 'end': prev_friday })
        if 'longueduree'    in config['main']['report_range'].split(','):
            date_ranges.append({ 'start': previous_monday_4Y, 'end': prev_friday })

    # store page content in a dictionary for bookmarks
    Bookmark = namedtuple('Bookmark', 'title page parent')
    bookmarks, i_page = [], 0
    bookmarks.append(Bookmark(title='Rapport pySchedVisu', page=i_page, parent=None))

    # create the multi-page report
    now_str = dt.now().strftime('%Y-%m-%d_%Hh%M')
    pdf_output_path = '{}/pySchedVisu_rapport_du_{}.pdf'.format(config['path']['output_dir'], now_str)
    with PdfPages(pdf_output_path) as pdf:

        # either go through all available machines, or use the list specified by the config
        machines_list = sorted(list(set([machine for machine in config['machines'].keys() if '_NoCT' not in machine])))
        if config['draw']['debug_single_machine'] != '*':
            machines_list = config['draw']['debug_single_machine'].split(',')

        # go through each machine
        for machine in machines_list:

            # create a parent entry for bookmarks
            bookmarks.append(Bookmark(title=machine, page=i_page, parent='Rapport pySchedVisu'))

            # go through each date range
            for date_range in date_ranges:

                # get the date ranges
                local_config = deepcopy(config)
                local_config['main']['start_date'] = date_range['start'].strftime('%Y%m%d')
                local_config['main']['end_date'] = date_range['end'].strftime('%Y%m%d')

                # create entry for bookmarks
                report_type = get_report_type(date_range['start'], date_range['end'])
                bookmarks.append(Bookmark(title=report_type, page=i_page, parent=machine))

                # create one page of the report
                create_page(local_config, machine)
                i_page += 1

                # save the page
                pdf.savefig()
                if not config['draw'].getboolean('debug_save_as_image'):
                    plt.close()

        d = pdf.infodict()
        d['Title'] = 'Rapport pySchedVisu'
        d['Author'] = 'Balazs Laurenczy'
        d['Subject'] = 'Utilisation des machines PET & SPECT du CHUV'
        d['Keywords'] = 'PET SPECT CHUV pySchedVIsu'
        d['CreationDate'] = dt.today()
        d['ModDate'] = dt.today()

    # add bookmarks
    add_bookmarks(pdf_output_path, bookmarks)

    return pdf_output_path

def add_bookmarks(pdf_output_path, bookmarks):
    """
    Create a copy of the PDF file with the same name but including bookmarks
    Args:
        pdf_output_path (str):      the path where the report is stored
        bookmarks (list of list):   the 2D array specifying how to put the bookmarks
    Returns:
        None
    """

    # store the handles of the created bookmarks
    bookmark_handles = {}

    # define the input and output objects
    reader = PdfFileReader(open(pdf_output_path, 'rb'))
    writer = PdfFileWriter()
    # copy meta data
    metadata = reader.getDocumentInfo()
    writer.addMetadata(metadata)
    # go through the bookmarks
    i_page, parent_bookmark_handle = 0, None
    for bookmark in bookmarks:
        # if we encounter a page we did not copy yet, add it
        if i_page == bookmark.page:
            writer.addPage(reader.getPage(i_page))
            i_page += 1
        # if the bookmark has the previous bookmark as a parent
        if bookmark.parent is not None and bookmark.parent in bookmark_handles.keys():
            bookmark_handles[bookmark.title] = \
                writer.addBookmark(bookmark.title, bookmark.page, bookmark_handles[bookmark.parent])
        else:
            bookmark_handles[bookmark.title] = writer.addBookmark(bookmark.title, bookmark.page)

    # write out the file
    temp_path = pdf_output_path.replace('.pdf', '__withBM.pdf')
    with open(temp_path, 'wb') as out:
        writer.write(out)

    # overwrite report
    shutil.move(temp_path, pdf_output_path)

def create_page(config, machine):
    """
    Create one page of the report
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        machine (str):  a string specifying the currently processed machine
    Returns:
        None
    """

    # get the date ranges
    start_date, end_date, _ = main.get_day_range(config)

    # load the relevant studies
    logging.info("Reading in studies")
    df, _ = extract_data.load_transform_and_save_data_from_files(config)

    # get the data for the current machine
    if df is None or len(df) == 0:
        logging.error('No data for {} {} - {} at these dates.'.format(machine,
            start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))
        return

    # exclude some machines and do some grouping up
    df['Machine'] = df['Machine Group'].str.replace('_NoCT', '')
    df = df[df['Machine'] != 'mixed cases']

    # get the data for the current machine
    df_machine = df.query('Machine == @machine')
    if len(df_machine) == 0:
        logging.error('No data for {} {} - {} for this machine at these dates.'.format(machine,
        start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))
        return

    # get the report type string
    report_type = get_report_type(start_date, end_date)
    logging.warning('Creating report for {} for {} - {}: "{}"'.format(machine,
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), report_type))

    # create a matplotlib figure with the right aspect ratio
    fig = plt.figure(figsize=[8.27, 11.69], dpi=config['draw'].getint('dpi'))

    # create the report, section by section
    create_header(config, fig, machine)
    # create_notes(config, fig)
    create_schedule(config, fig, machine, df_machine)
    create_daily_table(config, fig, machine, df_machine)
    create_violin(config, fig, machine, df_machine)
    create_stat_table(config, fig, machine, df_machine)

    if config['draw'].getboolean('debug_save_as_image'):
        #plt.show()
        logging.info("Saving PDF file")
        im_output_path = '{}/output_{}_{}_{}_{}'.format(config['path']['output_dir'],
            machine.lower().replace(' ', ''), report_type.replace(' ', ''),
            start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
        fig.savefig(im_output_path + '.pdf', orientation='portrait', papertype='a4', format='pdf')
        fig.savefig(im_output_path + '.png', orientation='portrait', papertype='a4', format='png')

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
    start_date, end_date, _ = main.get_day_range(config)
    n_days = (end_date - start_date).days
    week_numbers = sorted(list(set([start_date.strftime('%V'), end_date.strftime('%V')])))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(start_date, end_date)
    logging.debug(f"Header content: {report_type}, {week_numbers_str}")

    # draw the header text with dates, etc.
    plt.rcParams["font.family"] = "monospace"
    fig.text(0.62, 0.97, "Rapport {}".format(report_type), fontsize=15)
    fig.text(0.62, 0.93, "Semaine{} {}".format('s' if len(week_numbers) > 1 else '', week_numbers_str), fontsize=25, fontweight='bold')
    fig.text(0.63, 0.89, "du {}".format(start_date.strftime("%d/%m/%Y")), fontsize=20)
    fig.text(0.63, 0.86, "au {}".format(end_date.strftime("%d/%m/%Y")), fontsize=20)

    # machine name
    if len(machine) > 10:
        fig.text(0.04, 0.83, 'Machine: ' + machine, fontsize=14)
    else:
        fig.text(0.04, 0.83, 'Machine: ' + machine, fontsize=18)

    im_machine_path = '{}/images/{}.png'.format(os.getcwd(), machine.lower().replace(' ', '')).replace('/', '\\')
    im_machine = plt.imread(get_sample_data(im_machine_path))
    im_machine_ax = fig.add_axes([0.40, 0.815, 0.21, 0.04], anchor='NE')
    im_machine_ax.imshow(im_machine)
    im_machine_ax.axis('off')

    # draw the schedVisu logo
    im_logo_path = '{}/images/schedvisu_logo_transp.png'.format(os.getcwd()).replace('/', '\\')
    im_log = plt.imread(get_sample_data(im_logo_path))
    im_logo_ax = fig.add_axes([0.04, 0.83, 0.28, 0.15], anchor='NE')
    im_logo_ax.imshow(im_log)
    im_logo_ax.axis('off')

    # draw the CHUV logo
    im_chuv_logo_path = '{}/images/chuv.png'.format(os.getcwd()).replace('/', '\\')
    im_chuv_log = plt.imread(get_sample_data(im_chuv_logo_path))
    im_chuv_logo_ax = fig.add_axes([0.04, 0.75, 0.28, 0.15], anchor='NE')
    im_chuv_logo_ax.imshow(im_chuv_log)
    im_chuv_logo_ax.axis('off')

    # draw the VD logo
    im_vd_logo_path = '{}/images/vd.png'.format(os.getcwd()).replace('/', '\\')
    im_vd_log = plt.imread(get_sample_data(im_vd_logo_path))
    im_vd_logo_ax = fig.add_axes([0.02, 0.24, 0.02, 0.05], anchor='NE')
    im_vd_logo_ax.imshow(im_vd_log)
    im_vd_logo_ax.axis('off')

def get_report_type(start_date, end_date):
    """
    Return the report type ('hebdomadaire', 'bimensuel', 'mensuel', etc.) based on the week numbers provided.
    Args:
        start_date (datetime):  the starting date
        end_date (datetime):    the ending date
    Returns:
        report_type (string): a string representing the report type ('hebdomadaire', 'bimensuel', 'mensuel', etc.)
    """

    n_days = (end_date - start_date).days
    if   n_days < 1:                            report_type = 'journalier'
    elif n_days < 7:                            report_type = 'hebdomadaire'
    elif n_days < 14:                           report_type = 'bimensuel'
    elif n_days < 31:                           report_type = 'mensuel'
    elif n_days < 3 * 31:                       report_type = 'trimestriel'
    elif n_days < 6 * 31:                       report_type = 'semestriel'
    elif n_days < 365:                          report_type = 'annuel'
    else:                                       report_type = 'longue durée'

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

    logging.info("Creating schedule plot")
    create_schedule_plot(config, fig, machine, df)

    logging.info("Creating distribution plot")
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
    start_date, end_date, days_range = main.get_day_range(config)

    # create the new axes
    sched_ax = fig.add_axes([0.08, 0.42, 0.78, 0.39], anchor='NE')
    sched_ax.invert_yaxis()

    # create the ticks and labels, with a reduced frequency
    _, _, days_range_xticks = main.get_day_range(config, reduce_freq=True)
    days_xticks, days_xtick_labels = [], []

    # plot each day
    i_day = 0
    n_days_to_show = len(days_range)
    for day in days_range:
        # plot the day
        plot_day_for_schedule_plot(config, sched_ax, machine, day, i_day, df)

        # add the label if needed
        if day in days_range_xticks:
            days_xticks.append(i_day)
            # if we are displaying on a daily basis
            if days_range_xticks.freq.name in ['B', 'W-MON']:
                days_xtick_labels.append(day.strftime('%d/%m'))
            # if we are displaying on a monthly basis (e.g. whole year report)
            elif days_range_xticks.freq.name in ['4W-MON', 'BMS']:
                days_xtick_labels.append(day.strftime('%m/%y'))
            # if we are displaying on a bigger range
            else:
                days_xtick_labels.append(day.strftime('%Y'))

        logging.debug(f'day = {day.strftime("%Y%m%d")}, i_day = {i_day}, ' +
            f'n_days_to_show = {n_days_to_show}, len(days_range) = {len(days_range)}')

        i_day += 1
        # if we are a Friday and it is not the last day of the range, increase the index to create a gap
        if day.weekday() == 4 and i_day != n_days_to_show:
            logging.debug(f'Adding an extra "spacer" day {i_day}')
            i_day += 1
            n_days_to_show += 1

    # get the number of days displayed
    n_days = i_day
    # set the ticks, labels and the limits of the plot
    start_hour = config['draw'].getint('sched_start_hour')
    end_hour = config['draw'].getint('sched_end_hour')
    plt.xticks(days_xticks, days_xtick_labels)

    # calculate the x limits
    if      n_days <= 5:    plt.xlim([-0.5, n_days - 0.5])
    elif    n_days <= 11:   plt.xlim([-0.8, n_days - 0.2])
    elif    n_days <= 23:   plt.xlim([-1.0, n_days + 0.0])
    elif    n_days <= 104:  plt.xlim([-1.3, n_days + 0.3])
    else:                   plt.xlim([-1.5, n_days + 0.5])

    # set the y limits
    _set_schedule_y_lims(config, df)
    # add grid lines
    plt.grid(True, 'major', 'y', linestyle=':', linewidth=1)
    sched_ax.set_axisbelow(True)

def _set_schedule_y_lims(config, df):
    """
    Set the schedule-related plots y limits.
    Args:
        config (dict):      a dictionary holding all parameters for generating the report (dates, etc.)
        df (DataFrame):     a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    # calculate the best y limits
    start_times = df['Start Time'].apply(lambda st: pd.to_datetime(st, format='%H%M%S'))
    end_times = df['End Time'].apply(lambda et: pd.to_datetime(et, format='%H%M%S'))
    start_hours = start_times.apply(lambda st: st.hour + st.minute / 60 + st.second / 3600)
    end_hours = end_times.apply(lambda et: et.hour + et.minute / 60 + et.second / 3600)
    end_hour = max(round(max(end_hours) + 0.5), config['draw'].getint('sched_end_hour') + 0.5)
    start_hour =  min(round(min(start_hours) - 0.5), config['draw'].getint('sched_start_hour') - 0.5)
    # create the y-ticks range
    yticks_range = range(round(start_hour), round(end_hour) + 1)
    # do not show the last tick if that last tick is the plot's last Y limit
    if yticks_range[-1] == end_hour:    yticks_range = yticks_range[:-1]
    # do not show the first tick if that first tick is the plot's first Y limit
    if yticks_range[0] == start_hour:   yticks_range = yticks_range[1:]
    # create the y-ticks and set the limits
    plt.yticks(ticks=yticks_range, labels=['{:02.0f}h'.format(i) for i in yticks_range])
    plt.ylim((end_hour, start_hour))

def plot_day_for_schedule_plot(config, sched_ax, machine, day, i_day, df):
    """
    Plot a single day in the schedule plot.
    Args:
        config (dict):      a dictionary holding all parameters for generating the report (dates, etc.)
        sched_ax (Axes):    the matplotlib axes object for drawing
        machine (str):      a string specifying the currently processed machine
        day (datetime):     the currently processed day
        i_day (int):        index of the current day where this day should be plotted
        df (DataFrame):     a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = main.get_day_range(config)
    n_days = (end_date - start_date).days
    # initialize some variables related to the dates
    day_str = day.strftime('%Y%m%d')

    # get the data for the current day and machine
    df_day = df.query('Date == "{}" & Machine == "{}"'.format(day_str, machine)).sort_values('Start Time')
    logging.debug('Found {} studies in day {} (day number: {})'.format(len(df_day), day_str, i_day))
    # abort if no data
    if len(df_day) == 0: return

    # go through each study found for this day
    for i_study in range(len(df_day)):
        study = df_day.iloc[i_study, :]

        # get the start time, end time and duration as hours with decimals
        start_hour = _as_hour(study['Start Time'])
        end_hour = _as_hour(study['End Time'])
        duration_hours = end_hour - start_hour

        # get the start time, end time and duration as hours with decimals
        start_prep_hour = _as_hour(study['Start Time Prep'])
        end_prep_hour = _as_hour(study['End Time Prep'])
        duration_prep_hours = end_prep_hour - start_prep_hour

        # if the duration is negative
        if duration_hours <= 0:
            logging.warning('Problem with study {} on day {}: duration is 0 or negative: {}'
                .format(study.name, day_str, duration_hours))
            continue

        # get the coordinates where the rounded rectangle for this study should be plotted
        box_w = config['draw'].getfloat('study_box_w')
        box_w_prep = config['draw'].getfloat('study_box_w_prep')
        x_shift = config['draw'].getfloat('study_x_shift')
        x = i_day - (box_w * 0.5) + (x_shift * (-1 if (i_study % 2 == 0) else 1))
        y, h = start_hour, duration_hours
        x_prep = i_day - (box_w_prep * 0.5) + (x_shift * (-1 if (i_study % 2 == 0) else 1))
        y_prep, h_prep = start_prep_hour, duration_prep_hours

        # check if we have an overlap issue
        if i_study > 0:
            end_prev_hour = _as_hour(df_day.iloc[i_study - 1, :]['End Time'])
            if start_hour <= end_prev_hour:
                logging.debug(('Problem with study ...{} on {} & day {}: start hour {:6.3f} is ' +
                    'before end hour of last study {:6.3f}').format('.'.join(study.name.split('.')[-2:]),
                    machine, day_str, start_hour, end_prev_hour))

            # only plot gaps if we are plotting less than 6 months
            if n_days < 180:
                end_prev_prep_hour = _as_hour(df_day.iloc[i_study - 1, :]['End Time Prep'])
                # check how long the gap was with previous study
                gap_duration_hour = start_prep_hour - end_prev_prep_hour
                gap_threshold = config['draw'].getfloat('gap_dur_minutes_' + machine.lower().replace(' ', ''))
                # if the duration between the last end and the current start is bigger than the threshold
                if gap_duration_hour * 60 >= gap_threshold:
                    end_hours = df_day['End Time'].apply(_as_hour)
                    if any([end_hour > end_prev_prep_hour and end_hour < start_prep_hour for end_hour in end_hours]):
                        logging.warning('Found gap but it is between an overlapping study and the next study, so skipping.')
                    else:
                        line_width = 2
                        if n_days > 40: line_width = 1
                        # plot a black line to show gaps
                        plt.plot([i_day, i_day], [start_hour - 0.2, end_prev_hour + 0.2],
                            color='black', linestyle='dashed', linewidth=line_width)

        # get the start and stop times rounded to the minute
        logging.debug('day {}, start {:5.2f} -> end {:5.2f}, duration: {:4.2f}, i_day: {}'
            .format(day_str, start_hour, end_hour, duration_hours, i_day))

        # define colors
        descr_list = list(config['description_' + machine.lower().replace(' ', '')].keys()) + ['OTHER']
        colors = colors = config['draw']['colors'].split(',')
        i_descr = descr_list.index(study['Description'])

        # check if the current study is a retake
        try:
            i_take = int(study.name.split('_')[-1])
        except ValueError:
            logging.warning('Problem with study ...{} on {} & day {}: got a weird retake number: "{}"'
                .format('.'.join(study.name.split('.')[-2:]), machine, day_str,  study.name))
            i_take = 1

        hatch = ''
        if i_take != 1:
            logging.debug(study.name + ' is a retake (reprise)')
            hatch = '/'
            sibling_studies_patches = [
                    p for p in sched_ax.patches
                    if p._label.split('_')[0] == study.name.split('_')[0]\
                        and '_prep' not in p._label
                ]
            for p in sibling_studies_patches:
                p.set_hatch('\\')

        # if we are displaying more than ~4 months, the inside of the blocks is not visible anymore.
        #   Therefore, we need to use the edge to show the colors
        edge_color = 'black'
        if len(days_range) > 45:
            edge_color = colors[i_descr]
            hatch = ''

        if config['draw'].getboolean('debug_draw_prep_time'):
            # create the shape and plot it
            rounded_rect_prep = FancyBboxPatch((x_prep, y_prep), box_w_prep, h_prep,
                boxstyle="round,pad=-0.0040,rounding_size=0.155", fc=colors[i_descr], ec=edge_color, alpha=0.55,
                mutation_aspect=0.3, label=study.name + '_prep')
            sched_ax.add_patch(rounded_rect_prep)

            # create the shape and plot it
            rounded_rect = Rectangle((x, y), box_w, h, fc=colors[i_descr], ec=edge_color, hatch=hatch, label=study.name)
            sched_ax.add_patch(rounded_rect)

        else:
            # create the shape and plot it
            rounded_rect = FancyBboxPatch((x, y), box_w, h, boxstyle="round,pad=-0.0040,rounding_size=0.155",
                fc=colors[i_descr], ec=edge_color, mutation_aspect=0.3, hatch=hatch, label=study.name)
            sched_ax.add_patch(rounded_rect)

        # DEBUG show information string
        if config['draw'].getboolean('debug_schedule_show_IPP_string'):
            dur_str = '{}h{:02d}'.format(math.floor(duration_hours), int(duration_hours * 60 % 60))
            plt.text(x + box_w * 0.1, y + 0.9 * h, '{}: {} - {}: {}'
                .format(*study[['Patient ID', 'Start Time', 'End Time']], dur_str))

def _as_hour(t):
    """
    Returns the datetime object as hours.
    Args:
        t (str):        the time to convert to hours, formatted as %H%M%S
    Returns:
        t_h (float):    the time as hours
    """
    t_dt = pd.to_datetime(t, format='%H%M%S')
    return t_dt.hour + t_dt.minute / 60 + t_dt.second / 3600

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

    # check whether the distribution plot should be done or not
    start_date, end_date, days_range = main.get_day_range(config)
    if len(days_range) < 7 * 12: return

    # define some parameters for the distribution
    FMT = '%H%M%S'
    start_hour = '{:02d}3000'.format(config['draw'].getint('sched_start_hour') - 1)
    end_hour = '{:02d}3000'.format(config['draw'].getint('sched_end_hour'))
    # create a list of times which will be used as time points to build the distribution
    time_range = pd.date_range(dt.strptime(start_hour, FMT), dt.strptime(end_hour, FMT),
        freq=config['draw']['sched_distr_freq'])

    # define colors
    descr_list = list(config['description_' + machine.lower().replace(' ', '')].keys()) + ['OTHER']
    colors = colors = config['draw']['colors'].split(',')

    # add new axes
    distr_ax = fig.add_axes([0.86, 0.42, 0.12, 0.39], anchor='NE')
    distr_ax.invert_yaxis()

    logging.debug("Calculating distribution")
    for descr in descr_list[0:3]:
        # get all the studies for the selected time period
        df_period = df.query('Date >= "{}" & Date <= "{}" & Machine == "{}" & Description == "{}"'
            .format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), machine, descr))
        # initialize the DataFrame storing the counts for each time point
        df_counts = pd.DataFrame([0]*len(time_range), index=time_range, columns=['count'])

        # loop through each study
        for ind in df_period.index:
            # check if we have some leftover nans
            if str(df_period.loc[ind, 'Start Time']) == 'nan' or str(df_period.loc[ind, 'End Time']) == 'nan':
                logging.warning("Found some NaNs for study IPP{} on day {}: start = {}, end = {}"
                    .format(*df_period.loc[ind, ['Patient ID', 'Date', 'Start Time', 'End Time']]))
                continue
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
        # create new y values based on the time points
        counts_y_values = [t.hour + t.minute / 60 + t.second / 3600 for t in df_counts_resample.index]
        # plot the curve
        counts = df_counts_resample['count']
        if max(counts) == 0: continue
        plt.plot([count / max(counts) for count in counts], counts_y_values, color=colors[descr_list.index(descr)])

    # set the y limits
    _set_schedule_y_lims(config, df)

    # remove the ticks
    plt.xticks([])
    for tic in distr_ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    # add grid lines
    plt.grid(True, 'major', 'y', linestyle=':', linewidth=1)
    distr_ax.set_axisbelow(True)

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
    table_ax = fig.add_axes([0.08, 0.29, 0.78, 0.12], anchor='NE')
    table_ax.axis('off')

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = main.get_day_range(config, reduce_freq=True)

    # initialize the variable holding all the information to be displayed in the table
    data = [[],[],[],[],[]]
    cell_colors = [[],[],[],[],[]]
    # if we are not in a daily display mode, add a row for the number of days counting
    if days_range.freq.name != 'B':
        data.append([])
        cell_colors.append([])

    # go through each day
    i_day = 0
    for day in days_range:
        # get the next day in the range
        if len(days_range) > i_day + 1:
            next_day = days_range[i_day + 1] - timedelta(days=1)
            if next_day.weekday() == 6:
                next_day = days_range[i_day + 1] - timedelta(days=3)
        else:
            next_day = end_date
        i_day += 1

        # get the data and calculate values for the current day range
        df_day = df.query('Date >= "{}" & Date <= "{}" & Machine == "{}"'
            .format(day.strftime('%Y%m%d'), next_day.strftime('%Y%m%d'), machine))
        n_days_in_range = len(set(df_day['Date']))
        if n_days_in_range == 0:
            if machine != 'Millennium':
                logging.warning('Problem with day {}: n_days_in_range = {}, len(df_day) = {}!'
                    .format(day.strftime("%Y%m%d"), n_days_in_range, len(df_day)))
            n_days_in_range = 1
        n_max_studies_per_day = config['draw'].getint('n_study_per_day_' + machine.lower().replace(' ', ''))
        n_max_studies = n_max_studies_per_day * n_days_in_range
        n_studies = len(df_day)

        # count the number of gaps
        n_gaps = 0
        for i_study in range(len(df_day)):
            # get the start time, end time and duration as hours with decimals
            start_prep_hour = _as_hour(df_day.iloc[i_study]['Start Time Prep'])
            # check if we have an overlap issue
            if i_study > 0:
                # get the end time of the last study
                end_prev_prep_hour = _as_hour(df_day.iloc[i_study - 1]['End Time Prep'])
                # check how long the gap was with previous study
                gap_duration_hour = start_prep_hour - end_prev_prep_hour
                gap_threshold = config['draw'].getfloat('gap_dur_minutes_' + machine.lower().replace(' ', ''))
                if gap_duration_hour * 60 >= gap_threshold:
                    n_gaps += 1

        logging.debug('Processing day {}: next_day = {}, n_studies = {}, n_days_in_range = {}, n_max_studies = {}'
            .format(day.strftime("%Y%m%d"), next_day.strftime("%Y%m%d"), n_studies, n_days_in_range, n_max_studies))

        # if we are displaying on a daily basis
        if days_range.freq.name in ['B', 'W-MON']:
            if day == next_day:
                day_range_str = day.strftime("%d/%m")
            else:
                day_range_str = '{} - {}'.format(day.strftime("%d/%m"), next_day.strftime("%d/%m"))
        # if we are displaying on a monthly basis (e.g. whole year report)
        elif days_range.freq.name in ['4W-MON', 'BMS']:
            day_range_str = day.strftime("%m/%y")
        # if we are displaying on a bigger range
        else:
            day_range_str = day.strftime("%Y")

        # insert the values into the data table
        i_row = 0
        data[i_row].append(day_range_str)
        cell_colors[i_row].append('wheat')
        i_row += 1
        # if we are not in a daily display mode, add a row for the number of days counting
        if days_range.freq.name != 'B':
            data[i_row].append(n_days_in_range)
            cell_colors[i_row].append('w')
            i_row += 1
        data[i_row].append(n_max_studies)
        cell_colors[i_row].append('w')
        i_row += 1
        data[i_row].append(n_studies)
        cell_colors[i_row].append('w')
        i_row += 1
        data[i_row].append(n_gaps)
        cell_colors[i_row].append('w')
        i_row += 1
        data[i_row].append('{:3d}%'.format(int(100 * n_studies / n_max_studies)))
        cell_colors[i_row].append('w')

    headers = ['Date', 'Slot', 'Exam.', 'Trous', 'Util.']
    # if we are not in a daily display mode, add a row for the number of days counting
    if days_range.freq.name != 'B': headers.insert(1, 'Jours')
    header_colors = ['lightgray'] * len(headers)

    # plot the table
    table = plt.table(cellText=data, cellColours=cell_colors, rowLabels=headers,
        rowColours=header_colors, cellLoc='center', loc='center')
    table.set_fontsize(9)
    table.auto_set_font_size(False)

    # calculate summary values
    i_row = 0
    n_cols = len(data[i_row])
    i_row += 1
    # if we are not in a daily display mode, add a row for the number of days counting
    if days_range.freq.name != 'B':
        tot_days = sum([data[i_row][i_col] for i_col in range(n_cols) if isinstance(data[i_row][i_col], int)])
        i_row += 1
    tot_slots = sum([data[i_row][i_col] for i_col in range(n_cols) if isinstance(data[i_row][i_col], int)])
    i_row += 1
    tot_n_studies = sum([data[i_row][i_col] for i_col in range(n_cols) if isinstance(data[i_row][i_col], int)])
    i_row += 1
    tot_gaps = sum([data[i_row][i_col] for i_col in range(n_cols) if isinstance(data[i_row][i_col], int)])
    i_row += 1
    if tot_slots == 0: tot_slots = 0.01 # avoid crashing with a zero division
    # create the summary data table
    summ_data = []
    summ_data.append(['Total', 'Moyen'])
    summ_data.append([int(tot_slots), '{:.1f}'.format(tot_slots / n_cols)])
    summ_data.append([tot_n_studies, '{:.1f}'.format(tot_n_studies / n_cols)])
    summ_data.append([tot_gaps, '{:.1f}'.format(tot_gaps / n_cols)])
    summ_data.append(['', '{:.1f}'.format(100 * tot_n_studies / tot_slots)])
    summ_table_colors = [ ['lightgray']*2, ['w']*2, ['w']*2, ['w']*2, ['w']*2 ]

    # if we are not in a daily display mode, add a row for the number of days counting
    if days_range.freq.name != 'B':
        summ_data.insert(1, [int(tot_days), '{:.1f}'.format(tot_days / n_cols)])
        summ_table_colors = [ ['lightgray']*2, ['w']*2, ['w']*2, ['w']*2, ['w']*2, ['w']*2 ]

    # add new axes for the summary values
    table_summ_ax = fig.add_axes([0.86, 0.29, 0.12, 0.12], anchor='NE')
    table_summ_ax.axis('off')
    table_summ = plt.table(cellText=summ_data, cellColours=summ_table_colors, cellLoc='center', loc='center')
    table_summ.set_fontsize(10)
    table_summ.auto_set_font_size(False)

def create_violin(config, fig, machine, df):
    """
    Create the violin plot section.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    logging.info("Creating violin plot section")

    # add new axes
    vio_ax = fig.add_axes([0.09, 0.07, 0.40, 0.19], anchor='NE')

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = main.get_day_range(config)

    # get the list of descriptions for the currently processed machine
    descr_list = list(config['description_' + machine.lower().replace(' ', '')].keys()) + ['OTHER']
    colors = config['draw']['colors'].split(',')

    # define the variables storing the durations, the labels, etc.
    data, descr_names, x_positions = [], [], []

    # go through each study type
    for i_descr in range(len(descr_list)):
        # get the data and calculate values for the current day
        df_descr = df.query('Date >= "{}" & Date <= "{}" & Machine == "{}" & Description == "{}"'
            .format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), machine, descr_list[i_descr]))
        # calculate the durations
        if len(df_descr) > 0:
            start_times = pd.to_datetime(df_descr['Start Time'], format='%H%M%S')
            end_times = pd.to_datetime(df_descr['End Time'], format='%H%M%S')
            durations = (end_times - start_times).apply(lambda dur: dur.total_seconds() / 60)
            data.append(durations.values)
            descr_names.append(descr_list[i_descr])
            x_positions.append(i_descr)

    results = vio_ax.violinplot(data, x_positions, showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel('Durée (minutes)')
    plt.xticks(ticks=x_positions, labels=descr_names, rotation=60, fontsize=8)
    plt.xlim([-0.5, 9.5])

    # adjust the colors
    for i_body in range(len(results['bodies'])):
        results['bodies'][i_body].set_edgecolor(colors[x_positions[i_body]])
        results['bodies'][i_body].set_facecolor(_lighten_color(colors[x_positions[i_body]], 0.7))
        results['bodies'][i_body].set_alpha(1)
        results['cmeans'].set_color('black')
        results['cmeans'].set_alpha(0.4)
        results['cmins'].set_color('black')
        results['cmins'].set_alpha(0.4)
        results['cmaxes'].set_color('black')
        results['cmaxes'].set_alpha(0.4)
        results['cbars'].set_alpha(0)

def _lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Args:
        color (*):   can be matplotlib color string, hex string, or RGB tuple.
        amount (float): amount of lightening
    Returns:
        color (*):   the lightened color
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def create_stat_table(config, fig, machine, df):
    """
    Create the statistics table section with the statistics for each study type.
    Args:
        config (dict):  a dictionary holding all parameters for generating the report (dates, etc.)
        fig (Figure):   the matplotlib figure object for drawing
        machine (str):  a string specifying the currently processed machine
        df (DataFrame): a pandas DataFrame containing the studies to plot
    Returns:
        None
    """

    logging.info("Creating statistics table section")

    # add new axes
    table_ax = fig.add_axes([0.51, 0.02, 0.47, 0.34], anchor='NE')
    table_ax.axis('off')

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = main.get_day_range(config)

    # get the list of descriptions for the currently processed machine
    descr_list = list(config['description_' + machine.lower().replace(' ', '')].keys()) + ['OTHER']
    colors = config['draw']['colors'].split(',')

    # initialize the variable holding all the information to be displayed in the table
    data = [['DESCRIPTION', 'N.EXA', 'MOY', 'MIN', 'MAX']]
    cell_colors = [['lightgray'] + ['lightgray'] * 4]

    # go through each study type
    for i_descr in range(len(descr_list)):
        # append the row headers
        data.append([descr_list[i_descr]])
        cell_colors.append([colors[i_descr]])

        # get the data and calculate values for the current day
        df_descr = df.query('Date >= "{}" & Date <= "{}" & Machine == "{}" & Description == "{}"'
            .format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'), machine, descr_list[i_descr]))

        # append the number of studies
        data[i_descr + 1].append(len(df_descr))
        cell_colors[i_descr + 1].append('w')

        # if there is at least one study of the current type
        if len(df_descr) > 0:
            # calculate the durations
            start_times = pd.to_datetime(df_descr['Start Time'], format='%H%M%S')
            end_times = pd.to_datetime(df_descr['End Time'], format='%H%M%S')
            durations = end_times - start_times
            # add the mean, min and max values
            data, cell_colors = _add_cell_from_timedelta(data, cell_colors, durations.mean().total_seconds(), i_descr)
            data, cell_colors = _add_cell_from_timedelta(data, cell_colors, durations.min().total_seconds(), i_descr)
            data, cell_colors = _add_cell_from_timedelta(data, cell_colors, durations.max().total_seconds(), i_descr)

        # if there are no study of the current type
        else:
            logging.debug('Appending empty cells for {} because there is no data'.format(descr_list[i_descr]))
            data[i_descr + 1].extend(['-']*3)
            cell_colors[i_descr + 1].extend(['w']*3)

    # create the table and adjust its fontsize
    table = plt.table(cellText=data, cellLoc='center', loc='center',cellColours=cell_colors,
        colWidths=[0.45] + [0.1] + [0.15] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(7)

def _add_cell_from_timedelta(data, cell_colors, duration_seconds, i_descr):
    """
    Adds a cell to the statistics table.
    Args:
        data (list of list):        2-D array containing the cell's text content
        cell_colors (list of list): 2-D array containing the cell's color
        duration_seconds (int):     number of seconds to be displayed
        i_descr (int):              index of the current row
    Returns:
        data (list of list):        updated 2-D array containing the cell's text content
        cell_colors (list of list): updated 2-D array containing the cell's color
    """

    # transform the seconds into a MIN:SEC string
    duration_str = '{:02d}:{:02d}'.format(int(math.floor(duration_seconds / 60)), int(duration_seconds % 60))
    # append the value and the color
    data[i_descr + 1].append(duration_str)
    cell_colors[i_descr + 1].append('w')

    return data, cell_colors
