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
            logging.info('Reading   {}: save file found at "{}", loading data'.format(day_str, day_save_file_path))

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
        else:
            logging.info('Processing {}: no save file found at "{}"'.format(day_str, day_save_file_path))

    df_count_series = None
    if df_series is not None and len(df_series) > 0:
        # get a summary of what machines are used in which institution names and modality
        df_series, df_count_series, _ = do_series_groupby(config, df_series)

        # reset the index of the global DataFrame
        df_series = df_series.reset_index(drop=True)

    return df_series, df_count_series

def do_series_groupby(config, df_series_input):
    """
    Group series together using a defined set of columns.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame): a pandas DataFrame holding the series, annotated with the machine group info
        df_count_series (DataFrame): a pandas DataFrame holding the grouped series, counting the number of series
        df_count_studies (DataFrame): a pandas DataFrame holding the grouped series, counting the number of studies
    """

    # work with a copy that has a "fresh" index
    df_series = df_series_input.reset_index(drop=True)

    # drop the machine group list column to restart the processing from scratch
    df_series = df_series.drop(columns = ['Machine Group List', 'Machine Group'], errors = 'ignore')

    # clean up the institution name field
    df_series['Institution Name'] = df_series['Institution Name'].apply(lambda inst_name: inst_name.replace('  ', ' '))

    # rename the machine column name and "minify" the machine names
    df_series.loc[:, 'Machine short'] = df_series['Machine'].str.lower()
    df_series.loc[:, 'Machine short'] = df_series['Machine short'].apply(lambda m: re.sub(r'[ _]', '', m))

    # create a machine group name (as a comma-separated list) for each study
    df_machine_groups = df_series.groupby(['Study Instance UID'])['Machine short']
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
    df_series = df_series[~df_series['Machine'].isnull()]

    # aggregate series to count the number of series for each sub-group
    groupby_columns = ['Machine Group', 'Machine Group List', 'Institution Name', 'Machine', 'Modality']
    df_series_grouped_by_series = df_series.groupby(groupby_columns)
    df_count_series = pd.DataFrame(df_series_grouped_by_series.count())
    df_count_series = pd.DataFrame(df_count_series.rename(
        columns = {'Series Instance UID': 'Number of Series'})['Number of Series'])

    # aggregate series to count the number of studies for each sub-group
    df_series_grouped_by_study = df_series.groupby(groupby_columns + ['Study Instance UID'])
    df_count_studies = df_series_grouped_by_study.count().reset_index().groupby(groupby_columns).count()
    df_count_studies = pd.DataFrame(df_count_studies.rename(
        columns = {'Study Instance UID': 'Number of Studies'})['Number of Studies'])

    return df_series, df_count_series, df_count_studies

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
        df_last_series_for_study = df_series_for_study.sort_values('End Time')
        # there must be at least 2 series for any splitting
        if len(df_series_for_study) < 2: continue

        # check if there is any big difference in the successive series time
        time_diff = df_last_series_for_study['End Time'].apply(lambda t: dt.strptime(t, FMT)).diff()
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
                .format(study_str, split_indices[0], split_indices[0] - 1, df_series.loc[split_indices[0], 'End Time'],
                df_series.loc[split_indices[0] - 1, 'End Time'], time_diff_sec[split_indices[0]]))
            # mark the series according to the split index
            df_series.loc[[i for i in indices_for_study if i >= split_indices[0]], 'i_take'] = 2
            df_series.loc[[i for i in indices_for_study if i < split_indices[0]], 'i_take'] = 1

    return df_series
