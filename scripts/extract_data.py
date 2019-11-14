#!/usr/bin/env python

import logging
import os
import re
import pandas as pd
from datetime import datetime as dt


def extract_transform_and_save_data_from_files(config):
    """
    Extract, transform and save the relevant series for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    logging.info("Extracting data from files")

    # get the date range from the config
    start_date = dt.strptime(config['main']['start_date'], '%Y-%m-%d')
    end_date = dt.strptime(config['main']['end_date'], '%Y-%m-%d')
    days_range = pd.date_range(start_date, end_date)
    
    # create the variable holding all the series for all days
    df_series = None
    
    # go through the date range day by day
    for day in days_range:
        logging.debug('Processing {}'.format(day.strftime('%Y%m%d')))
        
        # create the path where the input day's data would be stored
        day_str = day.strftime('%Y%m%d')
        day_save_dir_path = os.path.join('data', day.strftime('%Y'), day.strftime('%Y-%m'))
        day_save_file_path = os.path.join(day_save_dir_path, '{}.pkl'.format(day.strftime('%Y-%m-%d')))
        
        # check if the current date has already been retrieved and saved
        if os.path.isfile(day_save_file_path):
            logging.info('Skipping   {}: save file found at "{}", loading data'.format(day_str, day_save_file_path))
            
            # load the data for the current day
            df_series_for_day = pd.read_pickle(day_save_file_path)
            # concatenate the series of the current day into the global DataFrame
            df_series = pd.concat([df_series, df_series_for_day], sort=False)
            
        # if the current date has not already been retrieved and saved
        else:
            logging.info('Processing {}: no save file found at "{}"'.format(day_str, day_save_file_path))
                    
    # get a summary of what machines are used in which institution names and modality
    df_groupby = do_series_groupby(config, df_series)
    logging.debug(df_groupby)

    return df_series, df_groupby


def do_series_groupby(config, df_series):
    """
    Group series together using a defined set of columns.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_grouped (DataFrame): a pandas DataFrame holding the grouped series
    """
    
    # define the columns to extract
    groupby_columns = ['Machine Name', 'Institution Name', 'Modality']
    # define the columns to use for the grouping
    extract_columns = ['machine', 'Institution Name', 'Study Instance UID', 'Modality']
  
    # create groups of machines
    machine_groups = []
    # create a DataFrame that regroups all the grouped sub-DataFrames
    df_grouped = pd.DataFrame()
    
    # get a list of machines and display it
    machines = set(df_series.machine[~df_series.machine.isnull()])
    logging.debug('Machines:')
    for machine in machines: logging.debug('  - {}'.format(machine))
    
    # go through the list of machines
    for machine in machines:
        # get all the study UIDs for this machine
        study_UIDs = set(df_series[df_series.machine == machine]['Study Instance UID'])
        # get the rows corresponding to these study UIDs
        rows_with_same_study_UIDs = df_series['Study Instance UID'].isin(study_UIDs)
        # extract some columns from these rows and do some renaming
        sub_df = df_series[rows_with_same_study_UIDs].loc[:, extract_columns]
        sub_df = sub_df.rename(columns = {'Study Instance UID': 'Number of Series'})
        sub_df = sub_df.rename(columns = {'machine': 'Machine Name'})
        
        # do the grouping using the input columns
        sub_df_grouped = sub_df.groupby(groupby_columns).count()
        
        # create a unique tag for the specific list of machine names of the selected rows
        machine_group_name = ', '.join(sorted(list(set(sub_df['Machine Name']))))
        # if the current list of machines already exists, do not repeat and continue
        if machine_group_name in machine_groups: continue
        
        # search the "common name" for this machine group using the config
        machine_group_common_name = 'undefined'
        short_machine_group_name = re.sub(r'[ _]', '', machine_group_name.lower())
        for conf_machine_name in config['machines']:
            logging.debug('Matching "{}" and "{}" ("{}") ...'
                .format(short_machine_group_name, config['machines'][conf_machine_name], conf_machine_name))
            if short_machine_group_name == config['machines'][conf_machine_name]:
                machine_group_common_name = conf_machine_name
                logging.debug('Found match between "{}" and "{}" ("{}")'
                    .format(machine_group_name, config['machines'][conf_machine_name], conf_machine_name))
                break
        
        # append the current machine group name to the list
        machine_groups.append(machine_group_name)
        # assign group index to the current machine group
        sub_df_grouped['Machine Group #'] = len(machine_groups)
        # assign group name to the current machine group
        sub_df_grouped['Machine Group Name'] = machine_group_common_name
        # append the current sub-DataFrame to the global DataFrame
        df_grouped = pd.concat([df_grouped, sub_df_grouped])

    # display the machine groups
    logging.debug('Machine groups:')
    for machine_group in machine_groups: logging.debug('  - {}'.format(machine_group))
    # sum up the counts
    df_grouped = df_grouped.groupby(['Machine Group #', 'Machine Group Name', 'Machine Name', 'Institution Name', 'Modality']).sum()
    
    return df_grouped


def mark_second_takes(config, df_series):
    """
    Mark series as being part of the first take (normal) or a second take (later in the day).
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
    df_series['i_take'] = 1

    # build the list of rows to keep
    indices_to_keep = []
    for i_study in range(len(study_UIDs)):
        
        # get the series related to the current Study Instance UID
        sUID = study_UIDs[i_study]
        study_str = '[{:2d}/{:2d}] {}...{}'.format(i_study, len(study_UIDs) - 1, sUID[:8], sUID[-5:])
        df_series_for_study = df_series[df_series['Study Instance UID'] == sUID]
        logging.info('Processing  {}: found {:2d} series'.format(study_str, len(df_series_for_study)))
        
        # extract the list of indices for the current study
        indices_for_study = list(df_series_for_study.index.values)
        # sort according to time and keep the last index
        df_last_series_for_study = df_series_for_study.sort_values('end_time')
        # there must be at least 2 series for any splitting
        if len(df_series_for_study) < 2: continue
        
        # check if there is any big difference in the successive series time
        time_diff = df_last_series_for_study['end_time'].apply(lambda t: dt.strptime(t, FMT)).diff()
        time_diff_sec = time_diff[1:].apply(lambda td: td.seconds)
        # get the DataFrame indices where the time difference is bigger than the threshold
        split_indices = list(time_diff_sec[time_diff_sec > study_split_thresh].index)
        
        # if there is no splitting indices
        if len(split_indices) == 0:
            logging.info('  Passing   {}: no second take (max time diff: {:4d} sec.)'.format(study_str, max(time_diff_sec)))
            continue
        
        # if there is more than one split point, throw an error and do not do any splitting
        elif len(split_indices) > 1:
            logging.error('  Error for {}: too many splitting points found: [{}]'
                          .format(study_str, ', '.join(str(split_indices))))
            df_series.loc[indices_for_study, 'i_take'] = None
            continue
        
        # if there is a single splitting time
        else:
            logging.info('  Splitting {}: split between {:3d}/{:3d} [T={}/{}, D={}]'
                .format(study_str, split_indices[0], split_indices[0] - 1, df_series.loc[split_indices[0], 'end_time'],
                df_series.loc[split_indices[0] - 1, 'end_time'], time_diff_sec[split_indices[0]]))
            # mark the series according to the split index
            df_series.loc[[i for i in indices_for_study if i >= split_indices[0]], 'i_take'] = 2
            df_series.loc[[i for i in indices_for_study if i < split_indices[0]], 'i_take'] = 1

    return df_series


def prune_by_time_overlap(df):
    """
    Prune the input DataFrame based on start/end time overlaps.
    Args:
        df (DataFrame): a pandas DataFrame to check for time overlaps
    Returns:
        df (DataFrame): the pruned pandas DataFrame
    """

    logging.info("Pruning DataFrame for time overlap")

    # time format
    FMT = '%H%M%S'

    # remove duplicates and sort (rows that have exactly the same start/end times are redundant)
    df = df.drop_duplicates(['start_time', 'end_time'])
    df = df[(~df['start_time'].isnull()) & (df['start_time'] != 'nan')]
    df = df.sort_values('start_time')

    # prune series based on start/end time overlaps:
    #   as long as some overlap was found, start over
    overlap_found = True
    while overlap_found:

        # make sure we only loop if an overlap was found, and reset the index of the DataFrame
        overlap_found = False

        # go through each row
        for i in range(1, len(df)):

            # get the start/end times of the current row
            curr_start = dt.strptime(df.iloc[i]['start_time'], FMT)
            curr_end = dt.strptime(df.iloc[i]['end_time'], FMT)
            # cget the start/end times of the previous row
            prev_start = dt.strptime(df.iloc[i - 1]['start_time'], FMT)
            prev_end = dt.strptime(df.iloc[i - 1]['end_time'], FMT)

            # check for an overlap between the current and the previous row
            latest_start = max(curr_start, prev_start)
            earliest_end = min(curr_end, prev_end)
            delta = (earliest_end - latest_start).seconds
            overlap = max(0, delta)

            # if there is no overlap, then the current time range is fully
            #   overlapping with the previous row's range
            overlap_found = overlap == 0
            logging.debug('{:2}/{}: checking if {}-{} is included in {}-{}: overlap = {}'.format(i, len(df) - 1,
                curr_start.strftime(FMT), curr_end.strftime(FMT), prev_start.strftime(FMT),
                prev_end.strftime(FMT), overlap_found))

            # if any overlap was found, remove the redundant row and start over
            if overlap_found:
                logging.info('{:2}/{}: found an overlap: {}-{} is included in {}-{}'.format(i, len(df) - 1,
                    curr_start.strftime(FMT), curr_end.strftime(FMT), prev_start.strftime(FMT),
                    prev_end.strftime(FMT)))
                df = df.drop(df.index[i])
                break

    return df
