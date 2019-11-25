#!/usr/bin/env python

import logging
import subprocess
import psutil
import signal
import time
import os
import shutil

from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
from pandas import DataFrame

import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom import dcmread

from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove
)

def retrieve_and_save_data_from_PACS(config):
    """
    Retrieve and save the relevant series from the PACS for all days specified by the config.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    logging.info("Retrieving data from PACS")

    # get the date range from the config
    start_date = dt.strptime(config['main']['start_date'], '%Y-%m-%d')
    end_date = dt.strptime(config['main']['end_date'], '%Y-%m-%d')
    days_range = pd.date_range(start_date, end_date)

    # go through the date range day by day
    for day in days_range:
        logging.debug('Processing {}'.format(day.strftime("%Y%m%d")))
        # fetch (or load) the data for the current day
        df_series_single_day = retrieve_and_save_single_day_data_from_PACS(config, day)

def retrieve_and_save_single_day_data_from_PACS(config, day):
    """
    Retrieve and save the relevant series from the PACS for a single day.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        day (datetime): a datetime object of the day to process
    Returns:
        None
    """

    # create the path where the input day's data would be stored
    day_str = day.strftime('%Y%m%d')
    day_save_dir_path = os.path.join('data', day.strftime('%Y'), day.strftime('%Y-%m'))
    day_save_file_path = os.path.join(day_save_dir_path, '{}.pkl'.format(day.strftime('%Y-%m-%d')))

    # check if the current date has already been retrieved and saved
    if os.path.isfile(day_save_file_path):
        logging.info('Skipping   {}: save file found at "{}", nothing to do'.format(day_str, day_save_file_path))

    # if the current date has not already been retrieved and saved
    else:
        logging.info('Processing {}: no save file found at "{}"'.format(day_str, day_save_file_path))
        # find all 'PT' and 'NM' studies for a day (specified as YYYYMMDD for the PACS)
        df_studies = find_studies_for_day(config, day.strftime('%Y%m%d'), ['PT', 'NM'])
        # abort if no studies provided as input
        if df_studies is None or len(df_studies) == 0:
            logging.warning('Warning at {}: no studies found'.format(day_str))
            return
        # get all series for the found studies
        df_series = find_series_for_studies(config, df_studies)
        # go through each series and find information about them
        df_series = fetch_info_for_series(config, df_series)
        # get some statistics on the success / failure rates of fetching info for SERIES
        show_stats_for_fetching_series_info(df_series)
        # get all series that have something wrong/missing
        df_failed_series = df_series[
            (df_series['Start Time'].isnull())
            | (df_series['End Time'].isnull())
            | (df_series['Machine'] == '')
            | (df_series['Institution Name'] == '')]
        # exclude series where some information could not be gathered (e.g. no end time or no machine)
        df_series = df_series.loc[df_failed_series.index, :]
        df_series = df_series.reset_index(drop=True)
        # make sure the save directory exists
        if not os.path.exists(day_save_dir_path): os.makedirs(day_save_dir_path)
        # save the series
        df_series.to_pickle(day_save_file_path)
        # save the failed series if required by the config
        if config['extract'].getboolean('debug_save_failed_series'):
            df_failed_series = df_failed_series.reset_index(drop=True)
            df_failed_series.to_pickle(day_save_file_path.replace('.pkl', '_failed.pkl'))

def find_studies_for_day(config, study_date, modality):
    """
    Finds all studies with given modality for a single day from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        study_date (str): a string specifying the day to query
        modality (str): a string specifying the modality to query ('PT', 'NM' or 'CT')
    Returns:
        df (DataFrame): a DataFrame containing all retrieved studies
    """

    logging.info('Processing {}: retrieving all studies from PACS'.format(study_date))

    # create the query dataset
    query_ds = Dataset()

    # parameters for filtering
    query_ds.QueryRetrieveLevel = 'STUDY'
    query_ds.ModalitiesInStudy = modality
    query_ds.StudyDate = study_date

    # parameters to fetch
    query_ds.StudyTime  = ''
    query_ds.StudyInstanceUID = ''
    query_ds.PatientID = ''
    query_ds.StudyDescription = ''
    query_ds.InstitutionName = ''

    # display the query dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # do the query (C-FIND)
    df_studies = find_data(config, query_ds)
    if df_studies is None or len(df_studies) == 0: return None

    # drop unwanted columns
    df_studies = df_studies.drop(config['retrieve']['to_drop_columns_studies'].split('\n'), axis=1, errors='ignore')

    # filter out irrelevant studies
    df_studies = df_studies[df_studies['Patient ID'].str.match('^\d+$')]
    df_studies = df_studies[~df_studies['Study Description'].isin(['EXTRINSEQUE'])]
    df_studies = df_studies.reset_index(drop=True)

    # DEBUGGING: in case a restriction on the number of studies should be done for faster processing (for debugging)
    n_max_studies = int(config['retrieve']['debug_n_max_studies_per_day'])
    if n_max_studies != -1: df_studies = df_studies.iloc[0 : n_max_studies, :]

    return df_studies

def find_series_for_studies(config, df_studies):
    """
    Finds all series for each study of the input DataFrame from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_studies (DataFrame): a pandas DataFrame specifying the studies to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved series for all studies
    """

    # this DataFrame stores the list of all the series found for all studies
    df_series = pd.DataFrame()

    # go through each study
    logging.info('Going through {} studie(s)'.format(len(df_studies)))
    for i_study in range(len(df_studies)):

        # find all series of the current study
        df_series_for_study = find_series_for_study(config, df_studies.iloc[i_study])
        if df_series_for_study is None:
            logging.warning('Skipping study because there are no usable Series associated with it')
            continue

        # DEBUGGING: in case a restriction on the number of studies should be done for faster processing (for debugging)
        n_max_series = int(config['retrieve']['debug_n_max_series_per_study'])
        if n_max_series != -1: df_series_for_study = df_series_for_study.iloc[0 : n_max_series, :]

        # get the list of valid/accepted institution names from the config
        accepted_inst_names = config['retrieve']['accepted_institution_names'].split('\n')
        df_series_for_study.loc[df_series_for_study['Institution Name'].isnull(), 'Institution Name'] = 'NONE'
        # get the institution name(s) for this study based on the found series
        inst_names = list(set([inst_name.replace('  ', ' ') for inst_name in df_series_for_study.loc[:, 'Institution Name']]))
        # if we found multiple institution names
        if len(inst_names) > 1:
            logging.warning('Multiple institution names for study: "{}"'.format(' / '.join(inst_names)))

            # check if any of these multiple institution names is valid / accepted
            if all([inst_name.lower().replace(' ', '') not in accepted_inst_names for inst_name in inst_names]):
                logging.warning('Skipping study because it is not from CHUV (but from "{}")'.format('" & "'.join(inst_names)))
                continue

            # if any of the institution name is valid, continue with a "mixed" institution name
            inst_name = 'mixed'

        # if we found a single institution name
        else:
            inst_name = inst_names[0]
        # set the institution name for this study
        df_studies.loc[i_study, 'Institution Name'] = inst_name

        # if this instiution name is not in the list of accepted institution names, skip it
        if inst_name.lower().replace(' ', '') not in accepted_inst_names:
            logging.warning('Skipping study because it is not from CHUV (but from "{}")'.format(inst_name))
            continue

        logging.debug('Appending {} series'.format(len(df_series_for_study)))
        # append the new series to the main series DataFrame
        df_series = df_series.append(df_series_for_study, sort=False, ignore_index=True)

    # add some required columns
    df_series['Start Time'] = None
    df_series['End Time'] = None
    df_series['Machine'] = None

    # DEBUGGING: in case a restriction on the number of studies should be done for faster processing (for debugging)
    n_max_series_per_day = int(config['retrieve']['debug_n_max_series_per_day'])
    if n_max_series_per_day != -1: df_series = df_series.iloc[0 : n_max_series_per_day, :]

    logging.info('Returning {} series'.format(len(df_series)))

    return df_series

def find_series_for_study(config, study_row):
    """
    Finds all series for a study from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        study_row (Series): a pandas Series (row) specifying the study to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved series for the queried study
    """

    # create an information string for the logging of the current series
    UID = study_row['Study Instance UID']
    study_string = '[{:2d}]: {}|{}|IPP:{:7s}|{}...{}'.format(study_row.name, *study_row[
        ['Study Date', 'Study Time', 'Patient ID']], UID[:8], UID[-4:])
    logging.info('Fetching series for {}'.format(study_string))

    # define some parameters
    series_level_filters = ['Study Date', 'Patient ID', 'Study Instance UID']
    to_drop_columns = ['Study Date', 'Query/Retrieve Level', 'Retrieve AE Title', 'Type of Patient ID',
        'Issuer of Patient ID']
    sort_columns = ['Series Time', 'Number of Series Related Instances']

    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(study_row, 'SERIES', incl=series_level_filters)

    # parameters for filtering
    query_ds.SeriesDate = query_ds.StudyDate
    query_ds.Modality = ['NM', 'PT', 'CT']

    # parameters to fetch
    query_ds.SeriesInstanceUID = ''
    query_ds.StudyDescription = ''
    query_ds.SeriesTime = ''
    query_ds.NumberOfSeriesRelatedInstances = ''
    query_ds.SeriesDescription = ''
    query_ds.ProtocolName = ''
    query_ds.InstitutionName = ''

    # display the query dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # do the query (C-FIND)
    df_series = find_data(config, query_ds)
    logging.debug('Found {} series before filtering description'.format(len(df_series)))
    # abort if no result
    if len(df_series) == 0: return None

    # filter out some Series that are not primary acquisitions (and do not contain any relevant time information)
    indices_to_exclude = []
    # go through each pattern and build the list of rows to exclude
    for descr_pattern in config['retrieve']['series_descr_patterns_to_exclude'].split('\n'):
        to_exclude_rows = df_series['Series Description'].str.match(descr_pattern, case=False)
        # gather all the indices
        indices_to_exclude.append(to_exclude_rows[to_exclude_rows == True].index)
    # flatten the list
    indices_to_exclude = [index for indices in indices_to_exclude for index in indices.values]
    # if there is something to exclude, show a message and drop the rows
    if len(indices_to_exclude) > 0:
        logging.debug('Found {} series to exclude based on their description: "{}"'.format(len(indices_to_exclude),
            '", "'.join(df_series.loc[indices_to_exclude]['Series Description'])))
        df_series.drop(indices_to_exclude, inplace=True)
    logging.debug('Found {} series after filtering description'.format(len(df_series)))
    # abort if no more result (all filtered)
    if len(df_series) == 0: return None

    # further filter out some Series that are not primary acquisitions (and do not contain any relevant time information)
    df_series = df_series[~df_series['Protocol Name'].isin(config['retrieve']['series_protocols_to_exclude'].split('\n'))]
    logging.debug('Found {} series after filtering protocol names'.format(len(df_series)))
    # abort if no more result (all filtered)
    if len(df_series) == 0: return None

    # drop unwanted columns, sort and display
    df_series = df_series.drop(config['retrieve']['to_drop_columns_series'].split('\n'), axis=1, errors='ignore')
    df_series.sort_values(sort_columns, inplace=True)
    df_series.reset_index(drop=True, inplace=True)

    return df_series

def fetch_info_for_series(config, df_series):
    """
    Get some information (start & end time, machine, etc.) for each series based on the images found in the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame): a pandas DataFrame holding the series
    """

    # fields to use as filters to get the image
    image_level_filters = ['Patient ID', 'Series Date', 'Series Instance UID', 'Modality']
    # fields to fetch from the DICOM header
    to_fetch_fields = ['SeriesInstanceUID', 'PatientID', 'InstanceNumber', 'ManufacturerModelName', 'AcquisitionTime',
        'Modality', 'ImageType', 'ActualFrameDuration', 'NumberOfFrames', '0x00540032', '0x00540052']

    # create the trial counting column if it does not exist yet
    if 'i_try' not in df_series.columns: df_series['i_try'] = None

    # go through each series several times
    for i_try in range(int(config['retrieve']['n_max_overall_try'])):

        # get the rows with missing info
        missing_rows = df_series['Start Time'].isnull()
        df_series.loc[missing_rows, 'i_try'] = i_try
        df_series_to_fetch = df_series[missing_rows].copy()

        # create the query datasets for each row
        query_datasets = df_series_to_fetch.apply(lambda row:
            create_dataset_from_dataframe_row(row, 'IMAGE', incl=image_level_filters, add_instance_number_filter=True),
            axis=1)
        logging.info('Created {} query datasets'.format(len(query_datasets)))

        # find information about this series by fetching some images
        df_info = get_data(config, query_datasets, to_fetch_fields)

        # process the fetched info and merge it back into the main df_series DataFrame
        df_series = process_and_merge_info_back_into_series(df_series, df_series_to_fetch, df_info)

    return df_series

def process_and_merge_info_back_into_series(df_series, df_series_to_fetch, df_info):
    """
    Calculate the end time of a NM series.
    Args:
        df_series (DataFrame): a pandas DataFrame holding the series
        df_series_to_fetch (DataFrame): a pandas DataFrame holding the series that do not have been fetched yet
        df_info (DataFrame): a pandas DataFrame holding the fetched info from the images
    Returns:
        end_time_str (str): the "End Time" as a string
    """

    # create masks for each modality
    pt, ct, nm = [df_info['Modality'] == modality for modality in ['PT', 'CT', 'NM']]
    # clean up the start times
    df_info.loc[:, 'AcquisitionTime'] = df_info.loc[:, 'AcquisitionTime'].apply(lambda t: str(t).split('.')[0])
    # convert the InstanceNumber field to an int (ignoring errors caused by empty values)
    df_info['InstanceNumber'] = df_info['InstanceNumber'].astype(int, errors='ignore')

    # create a main info storage DataFrame
    df_processed_info = DataFrame(columns=df_info.columns)

    ## Process CT and PT images
    df_ctpt = df_info[ct | pt].copy()
    if len(df_ctpt) > 0:
        # get the DataFrame as a grouped DataFrame counting the duplicated Series Instance UIDs (count() == 2)
        dups_count = df_ctpt.groupby('SeriesInstanceUID').count()['PatientID'] == 2
        # get the corresponding rows from the CT rows
        dups = df_ctpt[df_ctpt['SeriesInstanceUID'].isin(dups_count[dups_count].index.tolist())]
        # get the first and last instance of each duplicate
        first_images = dups[dups['InstanceNumber'] == 1]
        last_images = dups[dups['InstanceNumber'] > 1]
        # merge back both the first and last image into a single row
        df_ctpt = last_images.merge(first_images[['SeriesInstanceUID', 'PatientID', 'AcquisitionTime']], on=['SeriesInstanceUID', 'PatientID'])
        # check for inconsistencies: not all rows are duplicates
        if len(dups) != (len(df_ctpt) * 2):
            logging.warning('Missing some images, number of duplicates is not equal to the total number of images ({} != {} * 2)'
            .format(len(dups), len(df_ctpt)))
        # check for inconsistencies: not a perfect match of 1 first image for 1 last image
        if len(last_images) != len(first_images):
            logging.warning('Missing some images, number of first & last images is not equal ({} != {})'.format(len(last_images), len(first_images)))

        # rename the AcquisitionTime field from the last image as being the "End Time"
        df_ctpt = df_ctpt.rename(columns={'AcquisitionTime_x': 'Start Time', 'AcquisitionTime_y': 'End Time'})

        # append the CT and PT rows to the bigger DataFrame containing all modalities
        df_processed_info = df_processed_info.append(df_ctpt, sort=False)

    # Process NM images
    df_nm = df_info[nm].copy()
    if len(df_nm) > 0:
        # use the AcquisitionTime as Start Time
        df_nm['Start Time'] = df_nm['AcquisitionTime']
        # call a function to calculate the end times
        df_nm['End Time'] = df_nm.apply(get_NM_series_end_time, axis=1)

        # append the NM rows to the bigger DataFrame containing all modalities
        df_processed_info = df_processed_info.append(df_nm, sort=False)

    # if there is any resulting info
    if len(df_processed_info) > 0:
        # select the columns to merge back and rename them
        df_processed_info = df_processed_info[
            ['PatientID', 'SeriesInstanceUID', 'Start Time', 'End Time', 'ManufacturerModelName']]
        df_processed_info.columns = ['Patient ID', 'Series Instance UID', 'Start Time', 'End Time', 'Machine']
        # merge back the relevant info to this DataFrame
        df_series = df_series.merge(df_processed_info, on=['Patient ID', 'Series Instance UID'], how='outer')
        # clean up the columns, keeping only the values that are not null for each row
        for f in ['Start Time', 'End Time', 'Machine']:
            df_series[f] = df_series[f + '_y'].where(df_series[f + '_y'].notnull(), df_series[f + '_x'])
            df_series.drop(columns=[f + '_y', f + '_x'], inplace=True)

    return df_series

def get_NM_series_end_time(series_row):
    """
    Calculate the end time of a NM series.
    Args:
        series_row (Series): a pandas Series holding the series.
    Returns:
        end_time_str (str): the "End Time" as a string
    """

    # variable holding the duration of the current series
    series_duration = None

    # try to extract the "Phase Information Sequence"
    phase_sequence = series_row['0x00540032']
    # try to extract the "Rotation Information Sequence"
    rotation_sequence = series_row['0x00540052']

    if str(phase_sequence) != 'nan':
        # extract the duration of each "phase"
        phase_durations = []
        for phase in phase_sequence:
            frame_dur = int(phase['ActualFrameDuration'].value)
            n_frames = int(phase['NumberOfFramesInPhase'].value)
            phase_durations.append(frame_dur * n_frames)
        # calculate the sum of all durations and convert it to seconds 
        series_duration = sum(phase_durations)  / 1000
        logging.debug('  {}: duration is based on phase sequence'.format(series_row['SeriesInstanceUID']))

    elif str(rotation_sequence) != 'nan':
        # extract the duration of each "rotation"
        rotation_durations = []
        for rotation in rotation_sequence:
            frame_dur = int(rotation['ActualFrameDuration'].value)
            n_frames = int(rotation['NumberOfFramesInRotation'].value)
            rotation_durations.append(frame_dur * n_frames)
        # calculate the sum of all durations and convert it to seconds 
        series_duration = sum(rotation_durations)  / 1000
        logging.debug('  {}: duration is based on rotation sequence'.format(series_row['SeriesInstanceUID']))

    # if no "phase sequence vector" is present, use the actual frame duration
    elif str(series_row['ActualFrameDuration']) != 'nan' and str(series_row['NumberOfFrames']) != 'nan':
        # calculate the duration and convert it to seconds 
        series_duration = (int(series_row['ActualFrameDuration']) * series_row['NumberOfFrames']) / 1000
        logging.debug('  {}: duration is based on ActualFrameDuration'.format(series_row['SeriesInstanceUID']))

    # if a duration could *not* be extracted
    if series_duration is None:
        logging.error('  ERROR for {}: no series duration found'.format(series_row['SeriesInstanceUID']))
        return

    # if a duration info could be extracted, calculate the duration from the last instance's
    #   start time, as there could be multiple instances, even for a 'NM' series
    start_time_str = str(series_row['AcquisitionTime']).split('.')[0]
    start_time = dt.strptime(start_time_str, '%H%M%S')
    end_time = start_time + timedelta(seconds=series_duration)
    end_time_str = end_time.strftime('%H%M%S')

    return end_time_str

    df_series = df_series.drop(columns=['Machine', 'Start Time', 'End Time'], errors='ignore')
    # restore a "fresh" copy of the saved data
    df_info = df_info_save.copy()
    # create masks for each modality
    pt, ct, nm = [df_info['Modality'] == modality for modality in ['PT', 'CT', 'NM']]
    # clean up the start times
    df_info.loc[:, 'AcquisitionTime'] = df_info.loc[:, 'AcquisitionTime'].apply(lambda t: str(t).split('.')[0])

def show_stats_for_fetching_series_info(df_series):
    """
    Show some statistics on the fetching of information for seriess.
    Args:
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        None
    """

    try_level = 0
    try_col_name = 'i_try_' + str(try_level)
    # if we have the information about the first round
    while try_col_name in df_series.columns:

        # get what is the current total 
        if try_level == 0:
            n = len(df_series)
            successfull_prev_indices = []
        else:
            n = len(df_series[df_series['i_try_' + str(try_level - 1)] == -1])
            successfull_prev_indices = df_series[
                (df_series[
                    ['i_try_' + str(prev_try_level) for prev_try_level in range(0, try_level)]
                ] > 0)
                .apply(any, axis = 1)
            ].index.tolist()

        logging.info('Level {}: successfull_prev_indices = {}'.format(try_level, str(successfull_prev_indices)))
        # count successfull and failed trials on the first round
        i_try = df_series[try_col_name]
        failures = i_try[i_try == -1]
        logging.info('Level {}: failures = {}'.format(try_level, str(failures.index.tolist())))
        n_fail = len(failures)
        successes = i_try[i_try != -1]
        n_succ = len(successes)
        first = successes[successes == 1]
        multi = successes[successes > 1]

        # print out the stats for the first round
        logging.info('Failures      ({}): {:03d} / {:03d} ({:.1f}%)'
            .format(try_level, n_fail, n, 100 * n_fail / n))
        logging.info('Success       ({}): {:03d} / {:03d} ({:.1f}%)'
            .format(try_level, n_succ, n, 100 * n_succ / n))
        logging.info('  First tries ({}): {:03d} / {:03d} ({:.1f}%)'
            .format(try_level, len(first), n_succ, 100 * len(first) / n_succ))
        logging.info('  Multi-tries ({}): {:03d} / {:03d} ({:.1f}%)'
            .format(try_level, len(multi), n_succ, 100 * len(multi) / n_succ))
        logging.info('    Mean ± SD multi-tries ({}): {:.2f} ± {:.2f}'
            .format(try_level, multi.mean(), multi.std()))

        try_level += 1
        try_col_name = 'i_try_' + str(try_level)

def create_dataset_from_dataframe_row(df_row, qlevel, incl=[], excl=[], add_instance_number_filter=False):
    """
    Creates a pydicom Dataset for querying based on the content of an input DataFrame's row, provided
    as a Series.
    Args:
        df_row (Series): a DataFrame row containing all the information to create the query
        qrlevel (str): a string specifying the query (retrieve) level (PATIENT / STUDY / SERIES / IMAGE)
        incl (list): list of the columns to include in the Dataset. By default: all.
        excl (list): list of the columns to exlude in the Dataset. By default: none.
        add_instance_number_filter (bool): whether or not to add an InstanceNumber filter to the DataSet
    Returns:
        query_dataset (pydicom.dataset.Dataset): a Dataset object holding the filtering parameters
    """

    logging.debug("Creating dataset from row")

    # by default, include all columns
    if len(incl) == 0:
        incl = list(df_row.axes[0])

    logging.debug('List of columns: ' + ', '.join(incl))

    # create the Dataset with the appropriate query retrieve level
    ds = Dataset()
    ds.QueryRetrieveLevel = qlevel

    # go through each column and set them as attributes
    for col in incl:
        # do not include the columns present in the exclusion list
        if col in excl: continue
        # clean the name of the column
        col_name = col.title().replace(' ', '').replace('Uid', 'UID').replace('Id', 'ID')
        # get the value at that column
        value = df_row[col]
        # avoid NaNs
        if str(value) == 'nan': continue
        logging.debug('Setting "{}" ("{}") to "{}" (type: "{}")'.format(col, col_name, value, type(value)))
        setattr(ds, col_name, value)

    if add_instance_number_filter:
        # add some more filters for the 'PT' modality
        if df_row['Number of Series Related Instances'] != '1':
            image_number_filter = ['1', df_row['Number of Series Related Instances']]
        else:
            image_number_filter = '1'
        # add the instance number filters for the 'PT' & 'CT' modalities
        if df_row['Modality'] == 'PT' or df_row['Modality'] == 'CT':
            ds.InstanceNumber = image_number_filter

    return ds

def find_data(config, query_dataset):
    """
    Retrieve the data specified by the query dataset from the PACS using the C-FIND mode.
    Args:
        config (dict): a dictionary holding all the configuration parameters
        query_dataset (pydicom.dataset.Dataset): a Dataset object holding the filtering parameters
    Returns:
        df (DataFrame): a DataFrame containing all retrieved data
    """

    logging.debug("Finding data")

    # initialize a DataFrame to store the results
    df = DataFrame()
    # create the AE with the "find" information model
    ae = AE(ae_title=config['PACS']['local_ae_title'])
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    # associate with peer AE
    logging.debug("Connecting to PACS")
    assoc = ae.associate(config['PACS']['remote_host'], config['PACS'].getint('remote_port'),
                         ae_title=config['PACS']['remote_ae_title'])
    try:
        # if the connection is successfully established
        if assoc.is_established:
            logging.debug("Association established")
            # use the C-FIND service to send the identifier
            responses = assoc.send_c_find(query_dataset, PatientRootQueryRetrieveInformationModelFind)
            logging.debug("Response(s) received")
            i_response = 0
            for (status, identifier) in responses:
                if status:
                    logging.debug('C-FIND query status: 0x{0:04x}'.format(status.Status))
                    # if the status is 'Pending' then identifier is the C-FIND response
                    if status.Status in (0xFF00, 0xFF01):
                        # copy all fields
                        for data_element in identifier:
                            df.loc[i_response, data_element.name] = str(data_element.value)
                        i_response += 1
                else:
                    logging.error('Connection timed out, was aborted or received invalid response')
        else:
            logging.error('Association rejected, aborted or never connected')

    except:
        logging.error('Error during finding of data (C-FIND)')
        raise

    finally:
        # release the association
        assoc.release()
        logging.debug("Connection closed")

    return df
    
def get_data(config, query_dict, delete_data=True):
    """
    Retrieve the data specified by the query dictionary from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_df (DataFrame): a DataFrame containing one row for each series to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved data
    """

    # first download the DICOM files
    download_data_dcm4che(config, query_dict)
    # then read in the DICOM files
    df = read_DICOM_files(config)
    # if required, delete the DICOM files
    if delete_data:
        delete_DICOM_files(config)

def download_data_dcm4che(config, query_dict):
    """
    Download the data specified by the query dictionary from the PACS using the
        dcm4che toolkit (movescu & storescp exe files).
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_dict (dict): list of key-value pairs of the parameters for the query
    Returns:
        None
    """

    # tansform the dictionnary to a list of filtering parameters
    filter_params = []
    for key, item in query_dict.items():
        if isinstance(item, set):
            filter_params.extend(['-m', '{}={}'.format(key, '/'.join(item))])
        else:
            filter_params.extend(['-m', '{}={}'.format(key, item)])

    # create the command to launch the Store SCP server receiving the DICOMs
    storescp_commands = [
        '{dcm4che_path}/storescp.bat'.format(**config['retrieve'])
            .replace('/', '\\').replace('\\\\', '\\'),
        '-b', '{local_ae_title}@{local_host}:{local_port}'.format(**config['PACS']),
        '--directory', '{dicom_temp_dir}'.format(**config['retrieve'])
            .replace('/', '\\').replace('\\\\', '\\')
    ]

    # create the command to launch the MOVE SCU command to tell the PACS to send the DICOMs
    movescu_commands = [
        '{dcm4che_path}/movescu.bat'.format(**config['retrieve'])
            .replace('/', '\\').replace('\\\\', '\\'),
        '-c', '{remote_ae_title}@{remote_host}:{remote_port}'.format(**config['PACS']),
        '-b', '{local_ae_title}@{local_host}:{local_port}'.format(**config['PACS']),
        '--dest', '{local_ae_title}'.format(**config['PACS']),
        '-L', 'IMAGE'
    ] + filter_params
    
    # log the commands
    logging.debug(storescp_commands)
    logging.debug(movescu_commands)

    # catch errors because we want to make sure to kill the processes we spawn
    try:
    
        # launch both processes
        storescp_process = subprocess.Popen(storescp_commands)
        movescu_process = subprocess.Popen(movescu_commands)

        logging.debug('Created storescp process (PID={})'.format(storescp_process.pid))
        logging.debug('Created movescu process (PID={})'.format(movescu_process.pid))

        # wait until the C-MOVE is done
        while movescu_process.poll() is None:
            logging.debug('Waiting for C-MOVE to happen')
            time.sleep(0.5)
    # catch errors
    except:
        logging.error('Error while getting data using dcm4che')
        raise
        
    # make sure the processes are stopped
    finally:
    
        # kill the movescu process
        if movescu_process is not None:
            logging.debug('Killing movescu process')
            movescu_process.terminate()
            
        # kill the storescp process
        if storescp_process is not None:
            logging.debug('Killing storescp process')
            storescp_process.terminate()

        # the storescp process spawns a "java.exe" process that we need to manually find and kill
        find_and_kill_storescp_process()

def find_and_kill_storescp_process():
    """
    The storescp process spawns a "java.exe" process that we need to manually find and kill.
    Args:
        None
    Returns:
        None
    """

    logging.debug('Going through all processes to find the storescp "java.exe" process')
    # iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # check if the current process' name contains the searched name, and if there is a match,
            # check if the current java process contains the "storescp" string somewhere
            if 'java.exe'.lower() in proc.name().lower()\
                    and any(['storescp' in proc_cmd_part for proc_cmd_part in proc.cmdline()]):
                # if yes, kill it, kill it with fire
                logging.debug('Found storescp process and killing it')
                proc.kill()
                logging.debug('storescp process killed')

        # capture errors for unaccessible processes
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        # capture and display other errors
        except:
            logging.error('Problem killing storescp process')


def read_DICOM_files(config):
    """
    Read in as a DataFrame the data downloaded from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame): a DataFrame containing all the read data
    """

    DICOM_file_names = os.listdir(config['retrieve']['dicom_temp_dir'])
    logging.debug('Found {} DICOM file(s)'.format(len(DICOM_file_names)))
    to_fetch_fields = ['SeriesInstanceUID', 'PatientID', 'InstanceNumber', 'ManufacturerModelName', 'AcquisitionTime',
        'Modality', 'ImageType', 'ActualFrameDuration', 'NumberOfFrames', '0x00540032', '0x00540052']
    df = pd.DataFrame(columns=to_fetch_fields)
    i = 0
    for DICOM_file_name in DICOM_file_names:
        ds = dcmread(os.path.join(config['retrieve']['dicom_temp_dir'], DICOM_file_name),
            stop_before_pixels=True)
        for field in to_fetch_fields:
            logging.debug("Setting field '{}' for index {}".format(field, i))
            if Tag(field) in ds.keys():
                logging.debug("Setting field '{}' for index {} with value '{}'".format(field, i, ds[field].value))
                df.at[i, field] = ds[field].value
        i += 1

    return df 
   
def delete_DICOM_files(config):
    """
    Deletes the data downloaded from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    for filename in os.listdir(config['retrieve']['dicom_temp_dir']):
        file_path = os.path.join(config['retrieve']['dicom_temp_dir'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))