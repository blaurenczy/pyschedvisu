#!/usr/bin/env python

import logging
import os
import re
import pandas as pd
import numpy as np
import holidays
from copy import deepcopy
from datetime import timedelta
from datetime import datetime as dt

import main

def load_transform_and_save_data_from_files(config):
    """
    Load, transform and save the relevant series for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        df_studies (DataFrame): the pandas DataFrame holding the studies
    """

    # get the date range from the config
    start_date, end_date, days_range = main.get_day_range(config)
    # initialize the save path's location
    studies_save_path = config['path']['studies_db_save_path']
    series_save_path = config['path']['series_db_save_path']
    # initialize the studies and series DataFrames
    df_studies, df_studies_query, df_series, df_series_query = None, None, None, None
    # initialize the list of already_processed_days
    already_processed_days_studies, already_processed_days_series, already_processed_days = [], [], []
    # get the list of holiday days of Switzerland in the Canton de Vaud
    holiday_days = holidays.Switzerland(prov='VD', years=range(start_date.year, end_date.year + 1))
    for year in range(start_date.year, end_date.year + 1):
        holiday_days.append(dt(year, 12, 24))
        holiday_days.append(dt(year, 12, 26))
        holiday_days.append(dt(year, 12, 31))

    # check if the some studies have already been extracted
    if os.path.isfile(studies_save_path):
        logging.info('Reading studies database: save file found at "{}", loading data'.format(studies_save_path))
        # if yes, load them
        df_studies = pd.read_pickle(studies_save_path)

        # if specified by the config, filter for patient IDs
        if config['retrieve']['debug_patient_ids'] != '*':
            patientIDs = config['retrieve']['debug_patient_ids'].split(',')
            logging.warning('Restricting search to studies with patient IDs in [{}]'.format(','.join(patientIDs)))
            df_studies = df_studies.query('`Patient ID` in @patientIDs').copy()

        # get the list of the days that have already been processed
        already_processed_days_studies = list(set(df_studies['Date'].tolist()))

    # check if the some series have already been extracted
    if os.path.isfile(series_save_path):
        logging.info('Reading series database: save file found at "{}", loading data'.format(series_save_path))
        # if yes, load them
        df_series = pd.read_pickle(series_save_path)

        # if specified by the config, filter for patient IDs
        if config['retrieve']['debug_patient_ids'] != '*':
            patientIDs = config['retrieve']['debug_patient_ids'].split(',')
            logging.warning('Restricting search to series with patient IDs in [{}]'.format(','.join(patientIDs)))
            df_series = df_series.query('`Patient ID` in @patientIDs').copy()

        # get the list of the days that have already been processed
        already_processed_days_series = list(set(df_series['Date'].tolist()))

    # get the list of the days that are required by the config's day range but are not in the studies DataFrame
    already_processed_days = already_processed_days_studies + already_processed_days_series
    days_to_process = [day for day in days_range \
        if (day.strftime('%Y%m%d') not in already_processed_days \
        or config['extract'].getboolean('debug_force_extract_studies'))\
        and day.weekday() not in [5, 6] \
        and day not in holiday_days]

    # go through the days that need to be processed
    for day in days_to_process:
        if day.strftime('%Y%m%d') in already_processed_days \
            and config['extract'].getboolean('debug_force_extract_studies'):
            logging.info('Processing {} [{}/{}]: day is required and already processed but "force" option is on'
                .format(day.strftime('%Y%m%d'), days_to_process.index(day), len(days_to_process)))
        else:
            logging.warning('Processing {} [{}/{}]: day is required but not present in the main DataFrame'
                .format(day.strftime('%Y%m%d'), days_to_process.index(day), len(days_to_process)))

        # create a local config object just to process the specified days
        local_config = deepcopy(config)
        local_config['main']['start_date'] = day.strftime('%Y%m%d')
        local_config['main']['end_date'] = day.strftime('%Y%m%d')

        # load in the data
        df_series_for_day = load_data_from_files(local_config)
        if df_series_for_day is None or len(df_series_for_day) == 0: continue
        # mark the rektakes and the machine group for each series
        df_series_for_day = mark_machine_group(local_config, df_series_for_day)
        df_series_for_day = mark_retakes(local_config, df_series_for_day)

        # merge back the extracted series into the main DataFrame
        if df_series is None:
            df_series = df_series_for_day
        else:
            # remove any rows belonging to the same day, if any
            df_series = df_series[df_series['Date'] != day.strftime('%Y%m%d')]
            df_series = pd.concat([df_series, df_series_for_day], sort=False)\
                .sort_values(['Date', 'Start Time', 'Machine Group', 'SUID'])\
                .reset_index(drop=True)

        # group the series together into a DataFrame of studies
        df_studies_for_day = df_series_for_day.replace(np.nan, '').groupby('SUID').agg({
                'Date': lambda x: '/'.join(set(x)),
                'Start Time': 'min',
                'End Time': 'max',
                'Study Description': lambda x: '/'.join(set(x)),
                'Patient ID': lambda x: '/'.join(set(x)),
                'Machine Group': lambda x: '/'.join(set(x)),
                'Modality': lambda x: '/'.join(set(x)),
                'Protocol Name': lambda x: '/'.join(set(x))
            }).sort_values(['Start Time', 'Machine Group', 'SUID'])

        # create the description consensus
        df_studies_for_day = create_description_consensus(config, df_studies_for_day)
        df_studies_for_day = df_studies_for_day.sort_values(['Start Time', 'Machine Group', 'SUID'])
        # add the preparation times before and after each study
        df_studies_for_day = add_preparation_times(config, df_studies_for_day)
        df_studies_for_day = df_studies_for_day.sort_values(['Start Time', 'Machine Group', 'SUID'])
        # add the time to the previous and to the next study
        df_studies_for_day = add_time_to_prev_and_next(config, df_studies_for_day)

        # merge back the extracted studies into the main DataFrame
        if df_studies is None:
            df_studies = df_studies_for_day
        else:
            # remove any rows belonging to the same day, if any
            df_studies = df_studies[df_studies['Date'] != day.strftime('%Y%m%d')]
            df_studies = pd.concat([df_studies, df_studies_for_day], sort=False)\
                .sort_values(['Date', 'Start Time', 'Machine Group', 'SUID'])

    # abort if no studies could be loaded
    if df_studies is None or len(df_studies) == 0: return None, None

    # re-order the columns according to the config
    if df_series is not None and len(df_series) > 0:
        ordered_columns =  config['retrieve']['series_column_order'].split(',')
        unique_columns = set()
        add_to_unique_list = unique_columns.add
        columns = [
            col for col in ordered_columns + df_series.columns.tolist()
            if not (col in unique_columns or add_to_unique_list(col))
            and col in df_series.columns]
        df_series = df_series[columns]

    # if there was any change to the main DataFrame
    if len(days_to_process) > 0:
        df_studies = df_studies.drop_duplicates(['Start Time', 'End Time', 'Machine', 'Patient ID'])

        # save the updated DataFrame
        df_studies.to_pickle(studies_save_path)
        # save the updated DataFrame
        df_series.to_pickle(series_save_path)

    # create the query string to return the relevant studies and series
    query_str = 'Date >= "{}" & Date <= "{}"'.format(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
    logging.info(f'Query string: {query_str}')

    # get the relevant studies from the main studies DataFrame
    df_studies_query = df_studies.query(query_str).copy()
    logging.info('Returning {} studies from the total of {} studies'.format(len(df_studies_query), len(df_studies)))
    # get the relevant series from the main series DataFrame
    if df_series is not None and len(df_series) > 0:
        df_series_query = df_series.query(query_str).copy()
        logging.info('Returning {} series from the total of {} series'.format(len(df_series_query), len(df_series)))

    return df_studies_query, df_series_query

def load_data_from_files(config):
    """
    Load and return the relevant series for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        df_series (DataFrame): the pandas DataFrame holding the series
    """

    logging.debug("Loading data from files")

    # get the date range from the config
    start_date, end_date, days_range = main.get_day_range(config)
    # create the variable holding all the series for all days
    df_series = None

    # go through the date range day by day
    for day in days_range:
        logging.debug('Reading {}'.format(day.strftime('%Y%m%d')))

        try:

            # create the path where the input day's data would be stored
            day_save_dir_path = os.path.join(config['path']['data_dir'], day.strftime('%Y'), day.strftime('%Y-%m'))
            day_str = day.strftime('%Y%m%d')
            day_save_file_path = os.path.join(day_save_dir_path, '{}.pkl'.format(day_str))

            # check if the current date has already been retrieved and saved
            if os.path.isfile(day_save_file_path):
                logging.debug('Reading {}: save file found at "{}", loading data'.format(day_str, day_save_file_path))

                # load the data for the current day
                df_series_for_day = pd.read_pickle(day_save_file_path)
                # concatenate the series of the current day into the global DataFrame
                df_series = pd.concat([df_series, df_series_for_day], sort=False)

                # load the failed series if required by the config
                failed_day_save_file_path = day_save_file_path.replace('.pkl', '_failed.pkl')
                if config['extract'].getboolean('debug_load_failed_series') and os.path.isfile(failed_day_save_file_path):
                    logging.warning('Loading failed series for {}'.format(day_str))
                    df_failed_series = pd.read_pickle(failed_day_save_file_path)
                    # concatenate the failed series into the global DataFrame
                    df_series = pd.concat([df_series, df_failed_series], sort=False)

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

    if df_series is None: return None

    # if specified by the config, filter for patient IDs
    if config['retrieve']['debug_patient_ids'] != '*':
        patientIDs = config['retrieve']['debug_patient_ids'].split(',')
        logging.warning('Restricting search to series with patient IDs in [{}]'.format(','.join(patientIDs)))
        df_series = df_series.query('`Patient ID` in @patientIDs').copy()

    # create an index for the concatenated series
    df_series = df_series.reset_index(drop=True)

    # re-order the columns according to the config
    ordered_columns =  config['retrieve']['series_column_order'].split(',')
    unique_columns = set()
    add_to_unique_list = unique_columns.add
    columns = [
        col for col in ordered_columns + df_series.columns.tolist()
        if not (col in unique_columns or add_to_unique_list(col))
        and col in df_series.columns]
    df_series = df_series[columns]

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

    # remove duplicates
    df_series = df_series.drop_duplicates(['Patient ID', 'Start Time', 'End Time', 'Machine Group']).copy()
    df_series.reset_index(drop=True, inplace=True)

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

        # also check whether there is a series from another study inbetween our study:
        # get all the other series that are for the same day, same machine, but different Study Instance UID
        df_series_other = df_series[
            (df_series['Study Instance UID'] != sUID)
            & (df_series['Date'] == df_series_for_study.iloc[0]['Date'])
            & (df_series['Machine Group'] == df_series_for_study.iloc[0]['Machine Group'])]
        # get the start and end time of these other series
        start_times_other = pd.to_datetime(df_series_other['Start Time'], format=FMT)
        end_times_other = pd.to_datetime(df_series_other['Start Time'], format=FMT)
        # get the start and end time of the current study's series
        study_start = min(df_series_for_study['Start Time'])
        study_end = max(df_series_for_study['Start Time'])
        # get all the series that have their times inbetween our study
        df_series_inbetween = df_series_other[(start_times_other > study_start) & (end_times_other < study_end)]
        n_inbetween = len(df_series_inbetween)
        # if any inbetween series found
        if n_inbetween > 0:
            # get the start time of the inbetween series
            inbetween_start = min(pd.to_datetime(df_series_inbetween['Start Time'], format=FMT))
            # get the series from our study that splits our study (last series before the inbetween series)
            logging.info('Found {} series that are inbetween the start ({}) and end of study ({}): {}'
                .format(n_inbetween, study_start.strftime('%H:%M:%S'), study_end.strftime('%H:%M:%S'), study_str))
            new_series_split = df_series_for_study[df_series_for_study['Start Time'] > inbetween_start]\
                .sort_values('Start Time')
            df_series_split = df_series_split.append(new_series_split.iloc[0])

        # if there is no splitting indices
        if len(df_series_split) == 0:
            logging.debug('  Passing   {}: no second take (max time diff: {})'
                .format(study_str, max(df_series_for_study['time_to_prev'])))
            df_series.loc[df_series_for_study.index, 'i_take'] = 1
            continue

        # if there is more than one split point, throw an error and do not do any splitting
        elif len(df_series_split) >= 1:
            logging.debug('  Found {} series to split'.format(len(df_series_split)))
            # go through all the series
            i_take = 1
            for ind in df_series_for_study.index:
                if ind in df_series_split.index:
                    if ind <= 0:
                        logging.error('  Error at {}: trying to split at index "{}". Aborting.'
                            .format(study_str, ind))
                        continue
                    logging.debug('  Splitting {}: split {} between {:3d}/{:3d} [T={}/{}, D={}]'
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
    df_machine_groups = df_series.groupby('Study Instance UID')['Machine short']
    df_machine_groups = df_machine_groups.apply(lambda m: ','.join(sorted(list(set(m)))))
    df_machine_groups = df_machine_groups.reset_index().rename(
        columns = {'Machine short': 'Machine Group List'})

    # merge the machine group info back into the series DataFrame
    df_series = pd.merge(df_machine_groups, df_series, how='inner', on='Study Instance UID')

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

def create_description_consensus(config, df_studies):
    """
    Create a consensus on the studies description.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_studies (DataFrame): a pandas DataFrame holding the studies
    Returns:
        df_studies (DataFrame): a pandas DataFrame holding the studies, annotated with the description
    """

    # exclude some machines and do some grouping up
    df_studies = df_studies.copy()
    df_studies['Machine'] = df_studies['Machine Group'].str.replace('_NoCT', '')
    df_studies = df_studies[df_studies['Machine'] != 'mixed cases']

    # generate a minified version of the description without special characters or spaces
    df_studies['short_descr'] = df_studies['Study Description'].str.lower()\
                                    .apply(lambda m: re.sub(r'[ _\-^()\+:\.\']', '', m))

    # create a column holding the consensus descriptions
    df_studies['Description'] = None

    # go trough each machine
    for machine in set(df_studies['Machine']):
        # go through each description for the specified machine
        config_machine = config['description_' + machine.lower().replace(' ', '')]
        for descr in config_machine:
            # go through each description pattern for this description
            for descr_pattern in config_machine[descr].split(','):
                # get the matching studies
                df_studies_match = df_studies[
                    (df_studies['short_descr'].str.match('^' + descr_pattern + '$'))\
                    & (df_studies['Description'].isnull())\
                    & (df_studies['Machine'] == machine)]

                # set the consensus description for the matching studies
                df_studies.loc[df_studies.index.isin(df_studies_match.index), 'Description'] = descr

    df_studies.loc[df_studies['Description'].isnull(), 'Description'] = 'OTHER'
    df_studies = df_studies.drop(columns=['Protocol Name', 'short_descr'], errors='ignore')

    return df_studies

def add_preparation_times(config, df_studies):
    """
    Add preparation times for each study.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_studies (DataFrame): a pandas DataFrame holding the studies
    Returns:
        df_studies (DataFrame): a pandas DataFrame holding the studies, annotated with the preparation times
    """

    logging.info(f'Processing preparation times')

    # go through the studies machine by machine
    for machine in set(df_studies['Machine']):

        logging.info(f'Processing preparation times for {machine}')
        # get all studies for this machine
        df_machine = df_studies.query('Machine == @machine')
        # get the number of minutes to add before and after each study for the current machine
        field_name = 'prep_time_' + machine.lower().replace(' ', '')
        if field_name not in config['draw'].keys(): continue
        prep_time_sec = config['draw'].getint(field_name) * 60

        # go through the studies day by day
        for day in set(df_machine['Date']):
            logging.debug(f'Processing {machine} and {day}')
            # get all studies for this day
            df_day = df_machine.query('Date == @day').sort_values('Start Time')

            df_day['Start Time Prep'] = df_day['Start Time']\
                .apply(lambda st: dt.strptime(st, '%H%M%S') - timedelta(seconds=prep_time_sec))
            df_day['End Time Prep'] = df_day['End Time']\
                .apply(lambda et: dt.strptime(et, '%H%M%S') + timedelta(seconds=prep_time_sec))
            # go through each study to check whether there are overlaps
            for i in range(1, len(df_day)):
                logging.debug('Checking for overlap for {}, {}, {}:'.format(machine, day,
                    df_day.iloc[i]["Patient ID"]))
                logging.debug('Start current ({}) = {}, end previous ({}) = {}'
                    .format(i, df_day.iloc[i]['Start Time Prep'], i - 1, df_day.iloc[i - 1]['End Time Prep']))
                # if there is an overlap
                if df_day.iloc[i]['Start Time Prep'] < df_day.iloc[i - 1]['End Time Prep']:
                    # test whether it is a partial overlap (one study not fully contained in another)
                    if df_day.iloc[i - 1]['End Time Prep'] < df_day.iloc[i]['End Time Prep']:
                        # get the mid time in between the two studies
                        overlap_dur = df_day.iloc[i - 1]['End Time Prep'] - df_day.iloc[i]['Start Time Prep']
                        mid_time = df_day.iloc[i]['Start Time Prep'] + overlap_dur / 2
                        df_day.loc[df_day.iloc[i].name, 'Start Time Prep'] = mid_time
                        df_day.loc[df_day.iloc[i - 1].name, 'End Time Prep'] = mid_time
                        logging.debug(f'Partial overlap: {overlap_dur}, mid_time: {mid_time}')
                    # if it is a complete overlap, keep the original start and end times for the "included" study
                    else:
                        logging.debug(f'Complete overlap.')
                        df_day.loc[df_day.iloc[i].name, 'Start Time Prep'] \
                            = dt.strptime(df_day.iloc[i]['Start Time'], '%H%M%S')
                        df_day.loc[df_day.iloc[i].name, 'End Time Prep'] \
                            = dt.strptime(df_day.iloc[i]['End Time'], '%H%M%S')
            # transform datetimes to strings
            df_studies.loc[df_day.index, 'Start Time Prep'] \
                = df_day['Start Time Prep'].apply(lambda st: st.strftime('%H%M%S'))
            df_studies.loc[df_day.index, 'End Time Prep'] \
                = df_day['End Time Prep'].apply(lambda et: et.strftime('%H%M%S'))

    return df_studies



def add_time_to_prev_and_next(config, df_studies):
    """
    Add columns describing the time to the next and to the previous study.
    Args:
        config (dict):              a dictionary holding all the necessary parameters
        df_studies (DataFrame):     a pandas DataFrame holding the studies
    Returns:
        df_studies (DataFrame):     a pandas DataFrame holding the studies, annotated with the new columns
    """

    FMT = '%H%M%S'
    df = df_studies.copy()
    logging.info('Processing time-to-prev and time-to-next')

    # convert all the columns to datetime
    df['Start Time'] = pd.to_datetime(df['Start Time'], format=FMT)
    df['End Time'] = pd.to_datetime(df['End Time'], format=FMT)
    df['Start Time Prep'] = pd.to_datetime(df['Start Time Prep'], format=FMT)
    df['End Time Prep'] = pd.to_datetime(df['End Time Prep'], format=FMT)

    # compare the start time of a row with the end time of the previous row
    df['time_to_prev'] = df['End Time'].shift() - df['Start Time']
    df.loc[df['time_to_prev'] < timedelta(0), 'time_to_prev'] *= -1
    df['time_to_prev_prep'] = df['End Time Prep'].shift() - df['Start Time Prep']
    df.loc[df['time_to_prev_prep'] < timedelta(0), 'time_to_prev_prep'] *= -1

    # compare the end time of a row with the start time of the next row
    df['time_to_next'] = df['Start Time'].shift(-1) - df['End Time']
    df.loc[df['time_to_next'] < timedelta(0), 'time_to_next'] *= -1
    df['time_to_next_prep'] = df['Start Time Prep'].shift(-1) - df['End Time Prep']
    df.loc[df['time_to_next_prep'] < timedelta(0), 'time_to_next_prep'] *= -1

    # get the fully contained studies (studies that start after another study but finish before that same study)
    df['fully_contained'] = (df['End Time'] < df['End Time'].shift()) & (df['Start Time'] > df['Start Time'].shift())\
        & (df['Date'].shift() == df['Date']) & (df['Date'].shift(-1) == df['Date'])

    # make sure that we only keep values where the dates are identical
    df.loc[df['Date'] != df['Date'].shift(), 'time_to_prev'] = pd.NaT
    df.loc[df['Date'] != df['Date'].shift(), 'time_to_prev_prep'] = pd.NaT
    df.loc[df['Date'] != df['Date'].shift(-1), 'time_to_next'] = pd.NaT
    df.loc[df['Date'] != df['Date'].shift(-1), 'time_to_next_prep'] = pd.NaT

    # copy the columns
    for col in ['time_to_prev', 'time_to_prev_prep', 'time_to_next', 'time_to_next', 'fully_contained']:
        df_studies.loc[df.index, col] = df[col]

    return df_studies
