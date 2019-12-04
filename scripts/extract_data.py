#!/usr/bin/env python

import logging
import os
import re
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
from scripts.main import get_day_range

def load_transform_and_save_data_from_files(config):
    """
    Load, transform and save the relevant series for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        df_series (DataFrame): the pandas DataFrame holding the series
    """

    # create the path where the data for the current config would be stored
    day_period_str = '{}_{}'.format(config['main']['start_date'], config['main']['end_date']).replace('-', '')
    studies_save_path = 'data/studies/studies_{}.pkl'.format(day_period_str)

    # check if the data has already been extracted, transformed and saved
    if os.path.isfile(studies_save_path):
        logging.info('Reading {}: save file found at "{}", loading data'
            .format(day_period_str, studies_save_path))
        df_studies = pd.read_pickle(studies_save_path)

    # if the current date has not already been retrieved and saved
    else:
        logging.info('Reading {}: no save file found at "{}", processing data'
            .format(day_period_str, studies_save_path))

        # load in the data
        df_series = load_data_from_files(config)
        # mark the rektakes and the machine group for each series
        df_series = mark_retakes(config, df_series)
        df_series = mark_machine_group(config, df_series)
        # show some info about the series and studies
        show_series_groupby(config, df_series)

        # group the series together into a DataFrame of studies
        df_studies = df_series.dropna().groupby('SUID').agg({
            'Series Date': lambda x: '/'.join(set(x)),
            'Start Time': 'min',
            'End Time': 'max',
            'Study Description': lambda x: '/'.join(set(x)),
            'Patient ID': lambda x: '/'.join(set(x)),
            'Machine Group': lambda x: '/'.join(set(x)),
            'Modality': lambda x: '/'.join(set(x)),
            'Protocol Name': lambda x: '/'.join(set(x))
        }).sort_values(['Series Date', 'Start Time', 'Machine Group', 'SUID'])\
        .rename(columns={'Series Date': 'Date'})

        # save it to the studies file
        df_studies.to_pickle(studies_save_path)

    return df_studies

def load_data_from_files(config):
    """
    Load and return the relevant series for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        df_series (DataFrame): the pandas DataFrame holding the series
    """

    logging.info("Loading data from files")

    # get the date range from the config
    start_date, end_date, days_range = get_day_range(config)
    # create the variable holding all the series for all days
    df_series = None

    # go through the date range day by day
    for day in days_range:
        logging.debug('Reading {}'.format(day.strftime('%Y%m%d')))

        try:

            # create the path where the input day's data would be stored
            day_save_dir_path = os.path.join('data', day.strftime('%Y'), day.strftime('%Y-%m'))
            day_str = day.strftime('%Y%m%d')
            day_save_file_path = os.path.join(day_save_dir_path, '{}.pkl'.format(day.strftime('%Y-%m-%d')))

            # check if the current date has already been retrieved and saved
            if os.path.isfile(day_save_file_path):
                logging.debug('Reading {}: save file found at "{}", loading data'.format(day_str, day_save_file_path))

                # load the data for the current day
                df_series_for_day = pd.read_pickle(day_save_file_path)
                # concatenate the series of the current day into the global DataFrame
                df_series = pd.concat([df_series, df_series_for_day], sort=True)

                # load the failed series if required by the config
                failed_day_save_file_path = day_save_file_path.replace('.pkl', '_failed.pkl')
                if config['extract'].getboolean('debug_load_failed_series') and os.path.isfile(failed_day_save_file_path):
                    df_failed_series = pd.read_pickle(failed_day_save_file_path)
                    # concatenate the failed series into the global DataFrame
                    df_series = pd.concat([df_series, df_failed_series], sort=True)

            # if the current date has not already been retrieved and saved
            elif day.weekday() in [5, 6]:
                logging.debug('Reading {}: no save file for weekend "{}"'.format(day_str, day_save_file_path))

            # if the current date has not already been retrieved and saved
            else:
                logging.info('Reading {}: no save file found at "{}"'.format(day_str, day_save_file_path))

        except Exception as e:
            logging.error('Error while reading data for {}'.format(day.strftime("%Y%m%d")))
            logging.error("-"*60)
            logging.error(e, exc_info=True)
            logging.error("-"*60)

    # create an index for the concatenated series
    df_series = df_series.reset_index(drop=True)

    return df_series

def mark_retakes(config, df_series):
    """
    Mark series as being part of the first take (normal) or a re-take (new
        study with same patient later in the day, but with still the same Study Instance UID).
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame): the pandas DataFrame holding the marked series
    """

    # get the list of Study Instance UIDs
    study_UIDs = list(set(df_series['Study Instance UID']))
    logging.info('Found {} unique study UIDs'.format(len(study_UIDs)))

    FMT = '%H%M%S'
    # get from the config the threshold in seconds for splitting a study in "first/second take"
    study_split_thresh = int(config['extract']['n_sec_second_take_split_thresh'])

    # create a column to mark the "take" index of the series. By default, everything is a first take
    df_series['i_take'] = None

    # build the list of rows to keep
    indices_to_keep = []
    for i_study in range(len(study_UIDs)):

        # get the series related to the current Study Instance UID and sort them
        sUID = study_UIDs[i_study]
        df_series_for_study = df_series[df_series['Study Instance UID'] == sUID].sort_values('Start Time')

        study_str = '[{:4d}/{:4d}] {:52}'.format(i_study, len(study_UIDs) - 1, sUID)
        logging.debug('Processing  {}: found {:2d} series'.format(study_str, len(df_series_for_study)))

        # there must be at least 2 series for any splitting
        if len(df_series_for_study) < 2:
            df_series.loc[df_series_for_study.index, 'i_take'] = 1
            continue

        # convert the columns to datetime format
        df_series_for_study['Start Time'] = pd.to_datetime(df_series_for_study['Start Time'], format=FMT)
        df_series_for_study['End Time'] = pd.to_datetime(df_series_for_study['End Time'], format=FMT)
        # compare the start time of a row with the end time of the previous row
        df_series_for_study['time_to_prev'] = df_series_for_study['End Time'].shift() - df_series_for_study['Start Time']
        # correct for negative durations (when start time is before end time of previous row)
        df_series_for_study.loc[df_series_for_study['time_to_prev'] < timedelta(0), 'time_to_prev'] *= -1
        # get the series where a split should be done
        df_series_split = df_series_for_study[df_series_for_study['time_to_prev'] > timedelta(seconds=study_split_thresh)]

        # if there is no splitting indices
        if len(df_series_split) == 0:
            logging.debug('  Passing   {}: no second take (max time diff: {})'
                .format(study_str, max(df_series_for_study['time_to_prev'])))
            df_series.loc[df_series_for_study.index, 'i_take'] = 1
            continue

        # if there is more than one split point, throw an error and do not do any splitting
        elif len(df_series_split) >= 1:
            # go through all the series
            i_take = 1
            for ind in df_series_for_study.index:
                if ind in df_series_split.index:
                    logging.info('  Splitting {}: split {} between {:3d}/{:3d} [T={}/{}, D={}]'
                        .format(study_str, i_take, ind - 1, ind, df_series.loc[ind - 1, 'End Time'],
                        df_series.loc[ind, 'Start Time'], df_series_for_study.loc[ind, 'time_to_prev']))
                    i_take += 1
                # mark the series according to the split index
                df_series.loc[ind, 'i_take'] = i_take

    # create a new unique ID that includes the retake information
    df_series['SUID'] = df_series['Study Instance UID'] + '_' + df_series['i_take'].astype(str)

    return df_series

def mark_machine_group(config, df_series_input):
    """
    Add the machine group information to the series.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame): a pandas DataFrame holding the series, annotated with the machine group info
    """

    # work with a copy that has a "fresh" index, and no failed series
    df_series = df_series_input.reset_index(drop=True).copy()
    df_series = df_series[~df_series['Institution Name'].isnull()]
    df_series = df_series[df_series['Institution Name'] != '']
    df_series = df_series[df_series['Institution Name'] != 'NONE']
    df_series = df_series[~df_series['Machine'].isnull()]
    df_series = df_series[df_series['Machine'] != '']

    # drop the machine group list column to restart the processing from scratch
    df_series = df_series.drop(columns = ['Machine Group List', 'Machine Group'], errors = 'ignore')

    # clean up the institution name field
    df_series['Institution Name'] = df_series['Institution Name'].apply(lambda inst_name: inst_name.replace('  ', ' '))

    # rename the machine column name and "minify" the machine names
    df_series.loc[:, 'Machine short'] = df_series['Machine'].str.lower()
    df_series.loc[:, 'Machine short'] = df_series['Machine short'].apply(lambda m: re.sub(r'[ _]', '', m))

    # create a machine group name (as a comma-separated list) for each study
    df_machine_groups = df_series.groupby('SUID')['Machine short']
    df_machine_groups = df_machine_groups.apply(lambda m: ','.join(sorted(list(set(m)))))
    df_machine_groups = df_machine_groups.reset_index().rename(
        columns = {'Machine short': 'Machine Group List'})

    # merge the machine group info back into the series DataFrame
    df_series = pd.merge(df_machine_groups, df_series, how='inner', on='SUID')

    # create a DataFrame linking the machine group names to their comma-separated list
    conf_machine_group = pd.DataFrame(
        [ [conf_machine, config['machines'][conf_machine]] for conf_machine in config['machines'] ],
        columns = ['Machine Group', 'Machine Group List'])

    # merge the machine group name info back into the series DataFrame
    df_series = pd.merge(df_series, conf_machine_group, how='outer', on='Machine Group List')

    # mark the rows with an undefined machine group
    df_series.loc[df_series['Machine Group'].isnull(), 'Machine Group'] = 'mixed cases'

    # do some cleaning up
    df_series = df_series[~df_series['Machine'].isnull()]

    return df_series


def show_series_groupby(config, df_series):
    """
    Group series together using a defined set of columns.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        None
    """

    # aggregate series to count the number of series for each sub-group
    groupby_columns = ['Machine Group', 'Machine Group List', 'Institution Name', 'Machine', 'Modality']
    df_series_grouped_by_series = df_series.groupby(groupby_columns)
    df_count_series = pd.DataFrame(df_series_grouped_by_series.count())
    df_count_series = pd.DataFrame(df_count_series.rename(
        columns = {'Series Instance UID': 'Number of Series'})['Number of Series'])
    logging.info('Number of series for each sub-group')
    display(df_count_series)

    # aggregate series to count the number of studies for each sub-group
    df_series_grouped_by_study = df_series.groupby(groupby_columns + ['SUID'])
    df_count_studies = df_series_grouped_by_study.count().reset_index()
    df_count_studies = df_count_studies.groupby(groupby_columns).count()
    df_count_studies = pd.DataFrame(df_count_studies.rename(
        columns = {'SUID': 'Number of Studies'})['Number of Studies'])
    logging.info('Number of studies for each sub-group')
    display(df_count_studies)

    # aggregate series to count the number of series per day per machine
    groupby_columns = ['Series Date', 'Machine Group']
    df_series_grouped_by_series_day = df_series.groupby(groupby_columns)
    df_count_series_day = pd.DataFrame(df_series_grouped_by_series_day.count())
    df_count_series_day = pd.DataFrame(df_count_series_day.rename(
        columns = {'Series Instance UID': 'Number of Series'})['Number of Series'])
    df_count_series_day = df_count_series_day.unstack()
    logging.info('Number of series for each sub-group and each day')
    display(df_count_series_day)

    # aggregate series to count the number of studies per day per machine
    df_series_grouped_by_study_day = df_series.groupby(groupby_columns + ['SUID'])
    df_count_study_day = df_series_grouped_by_study_day.count().reset_index()
    df_count_study_day = df_count_study_day.groupby(groupby_columns).count()
    df_count_study_day = pd.DataFrame(df_count_study_day.rename(
        columns = {'Series Instance UID': 'Number of Studies'})['Number of Studies'])
    df_count_study_day = df_count_study_day.unstack()
    logging.info('Number of studies for each sub-group and each day')
    display(df_count_study_day)

    # aggregate series to count the number of studies per weekday per machine
    df_count_study_weekday = df_count_study_day.copy()
    df_count_study_weekday['Weekday'] = pd.Categorical([
        dt.strptime(d, '%Y%m%d').strftime("%A") for d in df_count_study_weekday.index],
        categories=['Monday','Tuesday','Wednesday','Thursday','Friday'], ordered=True)
    df_count_study_weekday = df_count_study_weekday.groupby('Weekday').sum()
    logging.info('Number of studies for each sub-group and each weekday')
    display(df_count_study_weekday)

    field_list = ['Institution Name', 'Machine', 'Machine Group', 'Modality',
        'Series Description', 'Study Description', 'Patient ID', 'i_take']
    for field in field_list:
        logging.info('Number of *Series* groupped by "{}"'.format(field))
        display(df_series.groupby(field)['SUID'].count())
        logging.info('Number of *Studies* groupped by "{}"'.format(field))
        display(df_series.groupby([field, 'SUID']).count().reset_index().groupby(field)['SUID'].count())
        logging.info('='*160)
