#!/usr/bin/env python

import logging
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mc
import colorsys
from random import random
from datetime import date, timedelta
from datetime import datetime as dt
import scripts.main
import scripts.extract_data

def create_report(config):
    """
    Generates a PDF report by extracting the relevant data from the database, processing it and drawing the relevant
    plots, tables and annotations.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    # load the relevant studies
    logging.info("Reading in studies")
    df = scripts.extract_data.load_transform_and_save_data_from_files(config)

    # exclude some machines and do some grouping up
    df['Machine'] = df['Machine Group'].str.replace('NoCT', '')
    df = df[df['Machine'] != 'mixed cases']

    start_date, end_date, _ = scripts.main.get_day_range(config)
    week_numbers = sorted(list(set([start_date.strftime('%V'), end_date.strftime('%V')])))
    week_numbers_str = '-'.join(week_numbers)
    report_type = get_report_type(week_numbers)

    # go through each machine
    #for machine in set(df['Machine']):
    for machine in ['PET Siemens']:

        # create a matplotlib figure with the right aspect ratio
        fig = plt.figure(figsize=[8.27, 11.69])

        # create the report, section by section
        create_header(config, fig, machine)
        #create_notes(config, fig)
        create_schedule(config, fig, machine, df)
        create_daily_table(config, fig, machine, df)
        create_violin(config, fig, machine, df)
        create_stat_table(config, fig, machine, df)

        logging.info("Saving PDF file")
        fig.savefig('output_{}_{}_{}_{}.pdf'
            .format(machine.lower().replace(' ', ''), report_type.replace(' ', ''),
            start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')),
            orientation='portrait', papertype='a4', format='pdf')

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
    logging.debug(f"Header content: {report_type}, {week_numbers_str}")

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
    im_machine_ax = fig.add_axes([0.40, 0.815, 0.21, 0.04], anchor='NE')
    im_machine_ax.imshow(im_machine)
    im_machine_ax.axis('off')

    ## draw the logo
    im_logo_path = '{}/images/logo_transp.png'.format(os.getcwd()).replace('/', '\\')
    im_log = plt.imread(get_sample_data(im_logo_path))
    im_logo_ax = fig.add_axes([0.01, 0.86, 0.58, 0.13], anchor='NE')
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
    start_date, end_date, days_range = scripts.main.get_day_range(config, remove_sundays=True)
    _, _, days_range_with_sundays = scripts.main.get_day_range(config)

    # create the new axes
    sched_ax = fig.add_axes([0.06, 0.42, 0.80, 0.39], anchor='NE')

    # plot each day
    for day in days_range:
        plot_day_for_schedule_plot(config, sched_ax, machine, day, df)

    # create the ticks and labels, with a reduced frequency
    _, _, days_range_ticks = scripts.main.get_day_range(config, remove_sundays=True, reduce_freq=True)
    days_xticks, days_xtick_labels = [], []
    for day in days_range_ticks:
        if str(days_range_ticks.freq) != '<MonthBegin>' and day.weekday() in [5, 6]: continue
        days_xticks.append(list(days_range_with_sundays).index(day) + 1)
        days_xtick_labels.append(day.strftime('%d/%m'))

    # set the ticks, labels and the limits of the plot
    start_hour = config['draw'].getint('sched_start_hour')
    end_hour = config['draw'].getint('sched_end_hour')
    plt.xticks(days_xticks, days_xtick_labels)
    plt.yticks(ticks=range(start_hour, end_hour + 1),
        labels=['{:02d}h'.format(i) for i in range(start_hour, end_hour + 1)])

    # calculate the x limits
    margin = len(days_range) / 14
    plt.xlim([max(1 - margin, 0.5), len(days_range) + margin])

    # calculate the best y limits
    #end_times = df['End Time'].apply(lambda et: pd.to_datetime(et, format='%H%M%S'))
    #end_hours = end_times.apply(lambda et: et.hour + et.minute / 60 + et.second / 3600)
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
    start_date, end_date, days_range = scripts.main.get_day_range(config, remove_sundays=True)
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
                logging.debug(('Problem with study ...{} on {} & day {}: start hour {:6.3f} is ' +
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

        # define colors
        descr_list = list(config['description_' + machine.lower().replace(' ', '')].keys()) + ['OTHER']
        colors = colors = config['draw']['colors'].split(',')
        i_descr = descr_list.index(study['Description'])

        # check if the current study is a retake
        i_take = int(study.name.split('_')[-1])
        hatch = ''
        edge_color = 'black'
        if i_take != 1:
            logging.debug(study.name + ' is a retake (reprise)')
            hatch = '/'
            edge_color = 'red'
            sibling_studies_patches = [
                p for p in sched_ax.patches if p._label.split('_')[0] == study.name.split('_')[0]]
            for p in sibling_studies_patches:
                p.set_hatch('\\')
                p.set_edgecolor('red')

        # create the shape and plot it
        rounded_rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=-0.0040,rounding_size=0.155",
            fc=colors[i_descr], ec=edge_color, mutation_aspect=0.3, hatch=hatch, label=study.name)
        sched_ax.add_patch(rounded_rect)

        # DEBUG show information string
        if config['draw'].getboolean('debug_schedule_show_IPP_string'):
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

    # check whether the distribution plot should be done or not
    start_date, end_date, days_range = scripts.main.get_day_range(config)
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
    # set some limits and remove the ticks
    start_hour = config['draw'].getint('sched_start_hour')
    end_hour = config['draw'].getint('sched_end_hour')
    plt.ylim((start_hour - 0.5, end_hour + 0.5))
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
    start_date, end_date, days_range = scripts.main.get_day_range(config, remove_sundays=True, reduce_freq=True)
    # initialize the variable holding all the information to be displayed in the table
    data = [[],[],[]]
    # go through each day
    i_day = 0
    for day in days_range:
        # get the next day in the range
        if len(days_range) > i_day + 1:
            next_day = days_range[i_day + 1]
        else:
            next_day = end_date
        i_day += 1
        # get the data and calculate values for the current day
        df_day = df.query('Date >= "{}" & Date < "{}" & Machine == "{}"'
            .format(day.strftime('%Y%m%d'), next_day.strftime('%Y%m%d'), machine))
        n_days = sum([1 for d in pd.date_range(day, next_day) if d.weekday() not in [5, 6]])
        n_max_studies = config['draw'].getint('n_study_per_day_' + machine.lower().replace(' ', '')) * n_days
        n_studies = len(df_day)
        # skip the weekends
        if str(days_range.freq) != '<MonthBegin>' and day.weekday() in [5, 6]:
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
    if tot_slots == 0: tot_slots = 0.01 # avoid crashing with a zero division
    summ_data = [
        [int(tot_slots), '{:.1f}'.format(tot_slots / n_cols)],
        [tot_n_studies, '{:.1f}'.format(tot_n_studies / n_cols)],
        ['', '{:.1f}'.format(100 * tot_n_studies / tot_slots)]]

    # add new axes for the summary values
    table_summ_ax = fig.add_axes([0.86, 0.31, 0.12, 0.12], anchor='NE')
    table_summ_ax.axis('off')
    table_summ = plt.table(cellText=summ_data, cellLoc='center', loc='center')
    table_summ.set_fontsize(10)
    table_summ.auto_set_font_size(False)

    # add new axes for the summary values
    table_summ_header_ax = fig.add_axes([0.86, 0.3885, 0.12, 0.02], anchor='NE')
    table_summ_header_ax.axis('off')
    table_summ_header = plt.table(cellText=[['Total', 'Moyen']], cellLoc='center', loc='center')
    table_summ_header.set_fontsize(10)
    table_summ_header.auto_set_font_size(False)

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
    vio_ax = fig.add_axes([0.09, 0.03, 0.40, 0.23], anchor='NE')

    # get the starting and ending dates, and the days range from the config
    start_date, end_date, days_range = scripts.main.get_day_range(config)

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
            durations = (end_times - start_times).apply(lambda dur: dur.total_seconds())
            data.append(durations.values)
            descr_names.append(descr_list[i_descr])
            x_positions.append(i_descr)

    results = vio_ax.violinplot(data, x_positions, showmeans=True, showextrema=True, showmedians=False)
    plt.ylabel('Durée (minutes)')
    plt.xticks(ticks=x_positions, labels=descr_names, rotation=75)
    y_minutes = range(0, 61, 10)
    plt.yticks(ticks=[x * 60 for x in y_minutes], labels=y_minutes)
    plt.ylim([(min(y_minutes) - 5) * 60, (max(y_minutes) + 5) * 60])

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
    start_date, end_date, days_range = scripts.main.get_day_range(config)

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
