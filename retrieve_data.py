#!/usr/bin/env python

# remove MATPLOTLIBDATA warning
import warnings
warnings.filterwarnings("ignore")

import logging
import subprocess
import psutil
import signal
import time
import os
import shutil
import holidays

from datetime import datetime as dt
from datetime import timedelta

import numpy as np
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

import main

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
    start_date, end_date, days_range = main.get_day_range(config)

    # go through the date range day by day
    for day in days_range:
        logging.debug('Processing {}'.format(day.strftime("%Y%m%d")))
        try:
            # fetch (or load) the data for the current day
            df_series_single_day = retrieve_and_save_single_day_data_from_PACS(config, day)

        except Exception as e:
            logging.error('Error while retrieving and saving data for {}'.format(day.strftime("%Y%m%d")))
            logging.error("-"*60)
            logging.error(e, exc_info=True)
            logging.error("-"*60)

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
    day_save_dir_path = os.path.join(config['path']['data_dir'], day.strftime('%Y'), day.strftime('%Y-%m'))\
        .replace('\\', '/')
    day_save_file_path = os.path.join(day_save_dir_path, '{}.pkl'.format(day.strftime('%Y%m%d'))).replace('\\', '/')

    # get the list of holiday days and add some custom off-days as well
    holiday_days = holidays.Switzerland(prov='VD', years=day.year)
    holiday_days.append(dt(day.year, 12, 31))
    holiday_days.append(dt(day.year, 12, 24))
    holiday_days.append(dt(day.year, 12, 26))

    # check if the current date has already been retrieved and saved
    if os.path.isfile(day_save_file_path):

        # if some failed series exist and they must be processed
        failed_day_save_file_path = day_save_file_path.replace('.pkl', '_failed.pkl')
        if os.path.isfile(failed_day_save_file_path) and config['retrieve'].getboolean('debug_recover_failed'):
            logging.warning('Processing   {}: save file found at "{}", but failed series exists and must be processed'\
                .format(day_str, day_save_file_path))
            # load both series (valid and failed)
            df_series = pd.read_pickle(day_save_file_path)
            df_failed_series = pd.read_pickle(failed_day_save_file_path)
            # concatenate the failed series into the global DataFrame
            df_series = pd.concat([df_series, df_failed_series], sort=True)
        # check if the current date has already been retrieved and saved but
        else:
            logging.info('Skipping   {}: save file found at "{}", nothing to do'.format(day_str, day_save_file_path))
            return

    # do not search for holiday days
    elif day in holiday_days:
        logging.info('Skipping   {}: save file not found at "{}", but this is a holiday.'
            .format(day_str, day_save_file_path))
        return

    # if the current date has not already been retrieved and saved, process it
    else:
        logging.warning('Processing {}: no save file found at "{}"'.format(day_str, day_save_file_path))

        # find all 'PT' and 'NM' studies for a day (specified as YYYYMMDD for the PACS)
        df_studies = find_studies_for_day(config, day.strftime('%Y%m%d'), ['PT', 'NM'])

        # if specified by the config, filter for patient IDs
        if config['retrieve']['debug_patient_ids'] != '*':
            patientIDs = config['retrieve']['debug_patient_ids'].split(',')
            logging.warning('Restricting search to studies with patient IDs in [{}]'
                .format(','.join(patientIDs)))
            df_studies = df_studies.query('`Patient ID` in @patientIDs').copy()

        # abort if no studies provided as input
        if df_studies is None or len(df_studies) == 0:
            logging.warning('Warning at {}: no studies found'.format(day_str))
            return

        # get all series for the found studies
        df_series = find_series_for_studies(config, df_studies)

    # check if there are any series found, otherwise abort
    if df_series is None or len(df_series) <= 0:
        logging.warning('Warning at {}: no series found'.format(day_str))
        return

    # if specified by the config, filter for patient IDs
    if config['retrieve']['debug_patient_ids'] != '*':
        patientIDs = config['retrieve']['debug_patient_ids'].split(',')
        logging.warning('Restricting search to studies with patient IDs in [{}]'
            .format(','.join(patientIDs)))
        df_series = df_series.query('`Patient ID` in @patientIDs').copy()

    # fetch the information for the series by batches from the PACS
    df_series, df_series_failed = fetch_info_for_series_with_batches(config, df_series)

    # make sure the save directory exists
    if not os.path.exists(day_save_dir_path): os.makedirs(day_save_dir_path)
    # save the series
    logging.info('Saving {} series'.format(len(df_series)))
    df_series.to_pickle(day_save_file_path)
    # save the failed series if there are some and if it is required by the config
    if len(df_series_failed) > 0 and config['retrieve'].getboolean('debug_save_failed_series'):
        df_series_failed = df_series_failed.reset_index(drop=True)
        df_series_failed.to_pickle(day_save_file_path.replace('.pkl', '_failed.pkl'))
        logging.info('Saving {} failed series'.format(len(df_series_failed)))

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
    # abort if there is no result or if some required columns are missing
    if df_studies is None or len(df_studies) == 0 or 'Study Description' not in df_studies.columns: return None

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
    logging.info('Finding all series for {} studie(s)'.format(len(df_studies)))
    df_studies = df_studies.reset_index(drop=True)
    for study_ind in df_studies.index:

        # find all series of the current study
        df_series_for_study = find_series_for_study(config, df_studies.loc[study_ind, :])
        if df_series_for_study is None or len(df_series_for_study) <= 0:
            logging.info('Skipping study because there are no usable Series associated with it')
            continue

        # DEBUGGING: in case a restriction on the number of studies should be done for faster processing (for debugging)
        n_max_series = int(config['retrieve']['debug_n_max_series_per_study'])
        if n_max_series != -1: df_series_for_study = df_series_for_study.iloc[0 : n_max_series, :]

        if 'Institution Name' not in df_series_for_study.columns:
            logging.info('Skipping study because there is no "Institution Name" information')
            continue

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
                logging.info('Skipping study because it is not from CHUV (but from "{}")'.format('" & "'.join(inst_names)))
                continue

            # if any of the institution name is valid, continue with a "mixed" institution name
            inst_name = 'mixed'

        # if we found a single institution name
        else:
            inst_name = inst_names[0]
        # set the institution name for this study
        df_studies.loc[study_ind, 'Institution Name'] = inst_name

        # if this instiution name is not in the list of accepted institution names, skip it
        if inst_name.lower().replace(' ', '') not in accepted_inst_names:
            logging.info('Skipping study because it is not from CHUV (but from "{}")'.format(inst_name))
            continue

        logging.debug('Study from {}: appending {} series'.format(inst_name, len(df_series_for_study)))
        # append the new series to the main series DataFrame
        df_series = df_series.append(df_series_for_study, sort=False, ignore_index=True)

    # add some required columns
    df_series['Start Time'] = None
    df_series['End Time'] = None
    df_series['Machine'] = None

    # re-order the columns according to the config
    ordered_columns =  config['retrieve']['series_column_order'].split(',')
    unique_columns = set()
    add_to_unique_list = unique_columns.add
    columns = [
        col for col in ordered_columns + df_series.columns.tolist()
        if not (col in unique_columns or add_to_unique_list(col))
        and col in df_series.columns]
    df_series = df_series[columns]

    # DEBUGGING: in case a restriction on the number of studies should be done for faster processing (for debugging)
    n_max_series_per_day = int(config['retrieve']['debug_n_max_series_per_day'])
    if n_max_series_per_day != -1: df_series = df_series.iloc[0 : min(len(df_series), n_max_series_per_day), :]

    logging.warning('Found {} series in total for the {} studies'.format(len(df_series), len(df_studies)))

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
    logging.debug('Fetching series for {}'.format(study_string))

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
    if df_series is None or len(df_series) == 0: return None

    # filter out some Series that are not primary acquisitions (and do not contain any relevant time information)
    indices_to_exclude = []
    # go through each pattern and build the list of rows to exclude
    if 'Series Description' in df_series.columns:
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
    # abort if no more result (all filtered) or if some required columns are missing
    if len(df_series) == 0 or 'Protocol Name' not in df_series.columns: return None

    # further filter out some Series that are not primary acquisitions (and do not contain any relevant time information)
    df_series = df_series[~df_series['Protocol Name'].isin(config['retrieve']['series_protocols_to_exclude'].split('\n'))]
    logging.debug('Found {} series after filtering protocol names'.format(len(df_series)))
    # abort if no more result (all filtered)
    if len(df_series) == 0: return None

    # keep only single instance NM series
    df_series_NM_multi_instance = df_series[(df_series['Modality'] == 'NM')\
        & (df_series['Number of Series Related Instances'].astype(int) > 1)]
    if len(df_series_NM_multi_instance) > 0:
        logging.warning('Discarding {} NM series because they have more than one instance'
            .format(len(df_series_NM_multi_instance)))
        df_series = df_series[~df_series.index.isin(df_series_NM_multi_instance)]
        # abort if no more result (all filtered)
        if len(df_series) == 0: return None

    # drop unwanted columns, sort and display
    df_series = df_series.drop(config['retrieve']['to_drop_columns_series'].split('\n'), axis=1, errors='ignore')
    df_series = df_series.rename(columns={'Series Date': 'Date'})
    df_series.sort_values(sort_columns, inplace=True)
    df_series.reset_index(drop=True, inplace=True)

    return df_series

def fetch_info_for_series_with_batches(config, df_series):
    """
    Fetch some information (start & end time, machine, etc.) for batches of series.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame):          a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame):          a pandas DataFrame holding the series
        df_series_failed (DataFrame):   a pandas DataFrame holding the failed series
    """

    # try to fetch the info for the series in an iterative way
    i_try = 0
    n_retry = config['retrieve'].getint('n_retry_per_day')
    n_series_per_batch = config['retrieve'].getint('n_series_per_batch')
    # make sure we try at least once
    if n_retry < 1: n_retry = 1
    for i_try in range(n_retry):
        # get all series that need to get more info
        df_series_no_info = df_series[
            (df_series['Start Time'].isnull())
            | (df_series['End Time'].isnull())
            | (df_series['Machine'] == '')
            | (df_series['Machine'].isnull())].copy()
        # process the series that need info as batches
        for i_series in range(0, len(df_series_no_info), n_series_per_batch):
            i_batch_start = i_series
            i_batch_end = min(len(df_series_no_info), i_series + n_series_per_batch)
            # limit the maximum number of series that is queried at once
            df_series_no_info_batch = df_series_no_info.iloc[i_batch_start:i_batch_end, :].copy()
            # go through each non-fetched series and find information about them
            logging.info('Fetching info for {} series, trial {}, batch {} - {}'
                .format(len(df_series_no_info_batch), i_try, i_batch_start, i_batch_end - 1))
            df_series_fetched, df_series_excl = fetch_info_for_series(config, df_series_no_info_batch)
            # get all series that have something wrong/missing from the series that were just fetched
            df_series_fetched_no_info = df_series_fetched[
                (df_series_fetched['Start Time'].isnull())
                | (df_series_fetched['End Time'].isnull())
                | (df_series_fetched['Machine'] == '')
                | (df_series_fetched['Machine'].isnull())]
            # exclude (from the fetched series DataFrame) all series that do not have info
            df_series_with_info = df_series_fetched.loc[
                ~df_series_fetched.index.isin(df_series_fetched_no_info.index), :]
            # exclude (from the main DataFrame) all series that were just fetched and that have some info
            df_series = df_series.loc[~df_series.index.isin(df_series_with_info.index), :]
            # add to the main DataFrame all the series that were just fetched and that have some info
            df_series = pd.concat([df_series, df_series_with_info], sort=True).reset_index(drop=True)

            df_series = df_series.drop_duplicates('Series Instance UID')
            # remove series that have the wrong image type from the list of series to fetch
            df_series = df_series[~df_series['Series Instance UID'].isin(df_series_excl['SeriesInstanceUID'])]

    # remove all duplicates
    df_series = df_series.drop_duplicates('Series Instance UID')
    # get all series that have something wrong/missing from the series
    df_series_failed = df_series[
        (df_series['Start Time'].isnull())
        | (df_series['End Time'].isnull())
        | (df_series['Machine'] == '')
        | (df_series['Machine'].isnull())
        | (df_series['Institution Name'] == '')
        | (df_series['Institution Name'] == 'NONE')
        | (df_series['Institution Name'].isnull())]
    # exclude (from the main DataFrame) all series that failed
    df_series = df_series.loc[~df_series.index.isin(df_series_failed.index), :]
    logging.warning('Found {} failed series and {} successful series'.format(len(df_series_failed), len(df_series)))

    # re-order the columns according to the config
    ordered_columns =  config['retrieve']['series_column_order'].split(',')
    unique_columns = set()
    add_to_unique_list = unique_columns.add
    columns = [
        col for col in ordered_columns + df_series.columns.tolist()
        if not (col in unique_columns or add_to_unique_list(col))
        and col in df_series.columns]
    df_series = df_series[columns]
    # sort the rows
    df_series.sort_values(['Date', 'Start Time', 'Patient ID', 'Machine', 'Series Description'], inplace=True)
    df_series.reset_index(drop=True, inplace=True)

    # reorder columns and sort the rows also for the failed series, if any
    if df_series_failed is not None and len(df_series_failed) > 0:
        df_series_failed = df_series_failed[columns]
        df_series_failed.sort_values(['Date', 'Start Time', 'Patient ID', 'Machine', 'Series Description'], inplace=True)
        df_series_failed.reset_index(drop=True, inplace=True)

    return df_series, df_series_failed

def fetch_info_for_series(config, df_series):
    """
    Get some information (start & end time, machine, etc.) for each series based on the images found in the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        df_series (DataFrame): a pandas DataFrame holding the series
    """

    # list of field names to extract
    to_fetch_fields = config['retrieve']['DICOM_tags_to_fetch'].split(',')

    # create subsets of the DataFrame based on the modality
    df_series_ctpt = df_series[df_series['Modality'].isin(['PT', 'CT'])]
    df_series_nm = df_series[df_series['Modality'] == 'NM']
    # initialize the variables to store the fetched info, which can remain empty
    df_info_ctpt = []
    df_info_nm = []

    # if there are some CT/PT rows to query
    if len(df_series_ctpt) > 0:
        # prepare the CT/PT queries for the first instance (first image)
        query_dicts_ctpt = list(df_series_ctpt.apply(lambda row: {
            'SeriesDate': row['Date'],
            'PatientID': row['Patient ID'],
            'SeriesInstanceUID': row['Series Instance UID'],
            'InstanceNumber': '1'
        }, axis=1))
        # prepare the CT/PT queries for the last instance (last image)
        df_last_frames = df_series_ctpt[df_series_ctpt['Number of Series Related Instances'] != '1']
        if df_last_frames is None or len(df_last_frames) > 0:
            query_dicts_ctpt.extend(
                df_last_frames.apply(lambda row: {
                    'SeriesDate': row['Date'],
                    'PatientID': row['Patient ID'],
                    'SeriesInstanceUID': row['Series Instance UID'],
                    'InstanceNumber': row['Number of Series Related Instances']
                }, axis=1))
        # fetch the CT/PT data
        logging.warning('Getting CT/PT data ({} queries)'.format(len(query_dicts_ctpt)))
        df_info_ctpt = get_data(config, query_dicts_ctpt, to_fetch_fields)

    # if there are some NM rows to query
    if len(df_series_nm) > 0:
        # prepare the NM queries for the first instance (first image)
        query_dicts_nm = list(df_series_nm.apply(lambda row: {
            'SeriesDate': row['Date'],
            'PatientID': row['Patient ID'],
            'SeriesInstanceUID': row['Series Instance UID']
        }, axis=1))
        # fetch the NM data
        logging.warning('Getting NM data ({} queries)'.format(len(query_dicts_nm)))
        df_info_nm = get_data(config, query_dicts_nm, to_fetch_fields)

    # process the fetched info and merge it back into the main df_series DataFrame
    df_series, df_series_to_exclude = process_and_merge_info_back_into_series(
                                        config, df_series, df_info_ctpt, df_info_nm)

    return df_series, df_series_to_exclude

def process_and_merge_info_back_into_series(config, df_series, df_info_ctpt, df_info_nm):
    """
    Process the fetched info for each series (get machine name, start & end time, etc.) and merge this info
        back into the main DataFrame (df_series).
    Args:
        config (dict):              a dictionary holding all the necessary parameters
        df_series (DataFrame):      a pandas DataFrame holding the series
        df_info_ctpt (DataFrame):   a pandas DataFrame holding the fetched info from the images for CT&PT
        df_info_nm (DataFrame):     a pandas DataFrame holding the fetched info from the images for NM
    Returns:
        df_series (str):            a pandas DataFrame holding the series
        df_series_to_exclude (str): a pandas DataFrame holding the series to exclude from search
    """

    # list of field names to extract
    to_fetch_fields = config['retrieve']['DICOM_tags_to_fetch'].split(',')

    # store series to exclude based on the image type
    df_series_to_exclude_ctpt, df_series_to_exclude_nm = pd.DataFrame(), pd.DataFrame()

    # Process PT/CT images
    if len(df_info_ctpt) > 0:

        # fix missing SeriesDate
        df_info_ctpt.loc[df_info_ctpt['SeriesDate'].isnull(), 'SeriesDate'] = \
            df_info_ctpt.loc[df_info_ctpt['SeriesDate'].isnull(), 'AcquisitionDate']
        df_info_ctpt = df_info_ctpt.drop(columns='AcquisitionDate')

        # check if there are any series to exclude based on the image type
        is_image_types_secondary_ctpt = df_info_ctpt['ImageType'].apply(str).str.match('.*SECONDARY.*')
        df_series_to_exclude_ctpt = df_info_ctpt[is_image_types_secondary_ctpt]
        if len(df_series_to_exclude_ctpt) > 0:
            logging.warning('Found {} CT/PT series to exclude based on the Image Type for day {}:'
                .format(len(df_series_to_exclude_ctpt), df_series_to_exclude_ctpt.iloc[0]['SeriesDate']))
            df_info_ctpt = df_info_ctpt[~is_image_types_secondary_ctpt].copy()
            logging.warning('  [Series Descriptions]:')
            for descr in df_series_to_exclude_ctpt['SeriesDescription'].tolist():
                logging.warning(f'    - "{descr}"')

        # get the images with a single instance
        single_instances_UIDs = df_series.loc[
            (df_series['Series Instance UID'].isin(df_info_ctpt['SeriesInstanceUID']))\
            & (df_series['Number of Series Related Instances'] == '1'), 'Series Instance UID']
        # duplicated them into the info DataFrame, so that they can also be merged together, as if there was two frames
        df_info_ctpt_single_inst = df_info_ctpt[df_info_ctpt['SeriesInstanceUID'].isin(single_instances_UIDs)].copy()
        df_info_ctpt_single_inst['InstanceNumber'] = 999999
        df_info_ctpt_extended = pd.concat([df_info_ctpt, df_info_ctpt_single_inst], sort=True)

        # clean up the start times
        df_info_ctpt_extended.loc[:, 'AcquisitionTime'] = df_info_ctpt_extended.loc[:, 'AcquisitionTime']\
            .apply(lambda t: str(t).split('.')[0])
        df_info_ctpt_extended.loc[:, 'ContentTime'] = df_info_ctpt_extended.loc[:, 'ContentTime']\
            .apply(lambda t: str(t).split('.')[0])

        # regroup the first and last instance rows on a single row
        df_info_ctpt_merged = df_info_ctpt_extended[df_info_ctpt_extended['InstanceNumber'].astype(int) == 1]\
            .merge(df_info_ctpt_extended[df_info_ctpt_extended['InstanceNumber'].astype(int) > 1],
                   on=['SeriesInstanceUID', 'SeriesDate', 'PatientID', 'ManufacturerModelName',
                   'Modality'], suffixes=['_start', '_end'])

        # rename the columns and keep the appropriate ones
        df_info_ctpt_clean = df_info_ctpt_merged.rename(columns={
                'SeriesInstanceUID': 'Series Instance UID',
                'PatientID': 'Patient ID',
                'ManufacturerModelName': 'Machine',
                'SeriesDate': 'Date',
                'AcquisitionTime_start': 'Start Time',
                'AcquisitionTime_end': 'End Time'})\
            .drop(columns=['InstanceNumber_start', 'InstanceNumber_end'])

        # make sure that ContentTime_start is before ContentTime_end for each rows, otherwise invert them
        s = pd.to_datetime(df_info_ctpt_clean['ContentTime_start'], format='%H%M%S')
        e = pd.to_datetime(df_info_ctpt_clean['ContentTime_end'], format='%H%M%S')
        df_inv = df_info_ctpt_clean[s > e].copy()
        df_inv[['ContentTime_start', 'ContentTime_end']] = df_inv[['ContentTime_end', 'ContentTime_start']]
        df_info_ctpt_clean[s > e] = df_inv

        # drop columns where all values are the same
        df_info_ctpt_clean_nonan = df_info_ctpt_clean.replace(np.nan, '')
        for f in to_fetch_fields:
            # if the _start and _end columns are both present
            if f + '_start' in df_info_ctpt_clean_nonan.columns and f + '_end' in df_info_ctpt_clean_nonan.columns:
                # check if all values are the same
                all_same = all(df_info_ctpt_clean_nonan[f + '_start'] == df_info_ctpt_clean_nonan[f + '_end'])
                logging.debug(f"Checking for duplicated columns for {f}: {all_same}")
                # if all values are the same, keep only one column
                if all_same:
                    df_info_ctpt_clean[f] = df_info_ctpt_clean[f + '_start']
                    df_info_ctpt_clean = df_info_ctpt_clean.drop(columns=[f + '_start', f + '_end'])

        # fix FDG Cerveau series where AcquisitionTime is not correct, therefore ContentTime needs to be used
        df_fdg_cerveau = df_info_ctpt_clean[df_info_ctpt_clean['SeriesDescription'].str.match('.*FDG Cerveau.*')]
        if len(df_fdg_cerveau) > 0:
            # get the start times of each FDG cerveau row
            start_times = pd.to_datetime(df_fdg_cerveau['Start Time'], format='%H%M%S')
            # get the number of seconds for each row
            duration_seconds = [int(60 * minutes) for minutes in df_fdg_cerveau['SeriesDescription']\
                .str.extract('.*FDG Cerveau (.+)min.*')[0].str.replace('x1','').astype(int).values]
            # calculate new end times
            end_times = [(start_times.iloc[i] + timedelta(seconds=duration_seconds[i])).strftime('%H%M%S')
                for i in range(len(start_times))]
            df_info_ctpt_clean.loc[df_fdg_cerveau.index, 'End Time'] = end_times

        # remove non-informative columns (all NaNs)
        df_info_ctpt_clean = df_info_ctpt_clean.dropna(how='all', axis=1)
        if len(df_info_ctpt_clean) > 0:

            df_info_ctpt_clean = df_info_ctpt_clean.drop(columns='SeriesDescription')

            # make sure that Start Time is before End Time for each rows, otherwise invert them
            s = pd.to_datetime(df_info_ctpt_clean['Start Time'], format='%H%M%S')
            e = pd.to_datetime(df_info_ctpt_clean['End Time'], format='%H%M%S')
            df_inv = df_info_ctpt_clean[s > e].copy()
            df_inv[['Start Time','End Time']] = df_inv[['End Time','Start Time']]
            df_info_ctpt_clean[s > e] = df_inv

            # merge the info into the series DataFrame
            df_series = df_series.merge(df_info_ctpt_clean, on=['Patient ID', 'Date', 'Series Instance UID', 'Modality'],
                how='outer')

            # keep only the relevant columns
            columns = df_series.columns
            columns_y = [col_y for col_y in columns if col_y[-2:] == '_y' and col_y.replace('_y', '_x') in columns]
            for col_y in columns_y:
                col = col_y.replace('_y', '')
                col_x = col_y.replace('_y', '_x')
                df_series[col] = df_series[col_y].where(df_series[col_y].notnull(), df_series[col_x])
                df_series.drop(columns=[col_y, col_x], inplace=True)

            # keep only the relevant columns
            columns = df_series.columns
            columns_s = [col_s for col_s in columns if col_s[-6:] == '_start' and col_s.replace('_start', '_end') in columns]
            for col_s in columns_s:
                col = col_s.replace('_start', '')
                col_end = col_s.replace('_start', '_end')
                df_series[col] = df_series[col_s].where(df_series[col_s].notnull(), df_series[col_end])
                df_series.drop(columns=[col_s, col_end], inplace=True)

    # Process NM images
    if len(df_info_nm) > 0:
        # fix missing SeriesDate
        df_info_nm.loc[df_info_nm['SeriesDate'].isnull(), 'SeriesDate'] = \
            df_info_nm.loc[df_info_nm['SeriesDate'].isnull(), 'AcquisitionDate']
        df_info_nm = df_info_nm.drop(columns='AcquisitionDate')

        # check if there are any series to exclude based on the image type
        is_image_types_secondary_nm = df_info_nm['ImageType'].apply(str).str.match('.*SECONDARY.*')
        df_series_to_exclude_nm = df_info_nm[is_image_types_secondary_nm]
        if len(df_series_to_exclude_nm) > 0:
            logging.warning('Found {} NM series to exclude based on the Image Type for day {}:'
                .format(len(df_series_to_exclude_nm), df_series_to_exclude_nm.iloc[0]['SeriesDate']))
            df_info_nm = df_info_nm[~is_image_types_secondary_nm].copy()
            logging.warning('  [Series Descriptions]:')
            for descr in df_series_to_exclude_nm['SeriesDescription'].tolist():
                logging.warning(f'    - "{descr}"')

        # clean up the start times
        df_info_nm.loc[:, 'AcquisitionTime'] = df_info_nm.loc[:, 'AcquisitionTime']\
            .apply(lambda t: str(t).split('.')[0])
        # use the AcquisitionTime as Start Time
        df_info_nm['Start Time'] = df_info_nm['AcquisitionTime']
        # call a function to calculate the End Times
        df_info_nm['End Time'] = df_info_nm.apply(get_NM_series_end_time, axis=1)
        # rename the columns
        df_info_nm_clean = df_info_nm.rename(columns={
                'SeriesInstanceUID': 'Series Instance UID',
                'PatientID': 'Patient ID',
                'SeriesDate': 'Date',
                'ManufacturerModelName': 'Machine',
                '0x00180050': 'SliceThickness',
                '0x00189931': 'scanPositionStart',
                '0x00189932': 'scanPositionEnd',
                '0x00181302': 'ScanLength',
                '0x00201002': 'ImagesInAcquisition' })\
            .drop(columns=['SeriesDescription', '0x00540032','0x00540052'])

        # merge the info into the series DataFrame
        df_series = df_series.merge(df_info_nm_clean, on=['Patient ID', 'Date', 'Series Instance UID', 'Modality'],
            how='outer')

        # keep only the relevant columns
        columns = df_series.columns
        columns_y = [col_y for col_y in columns if col_y[-2:] == '_y' and col_y.replace('_y', '_x') in columns]
        for col_y in columns_y:
            col = col_y.replace('_y', '')
            col_x = col_y.replace('_y', '_x')
            df_series[col] = df_series[col_y].where(df_series[col_y].notnull(), df_series[col_x])
            df_series.drop(columns=[col_y, col_x], inplace=True)

        # keep only the relevant columns
        columns = df_series.columns
        columns_s = [col_s for col_s in columns if col_s[-6:] == '_start' and col_s.replace('_start', '_end') in columns]
        for col_s in columns_s:
            col = col_s.replace('_start', '')
            col_end = col_s.replace('_start', '_end')
            df_series[col] = df_series[col_s].where(df_series[col_s].notnull(), df_series[col_end])
            df_series.drop(columns=[col_s, col_end], inplace=True)

    # merge together the excluded series from both modality groups
    df_series_to_exclude = pd.concat([df_series_to_exclude_ctpt, df_series_to_exclude_nm], sort=True)

    # remove duplicates
    df_series = df_series.drop_duplicates('Series Instance UID')

    return df_series, df_series_to_exclude

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

    try:
        # try to extract the "Phase Information Sequence"
        phase_sequence = series_row['0x00540032']
        # try to extract the "Rotation Information Sequence"
        rotation_sequence = series_row['0x00540052']

        if str(phase_sequence) != 'nan':
            logging.debug(phase_sequence)
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
            logging.debug(rotation_sequence)
            # extract the duration of each "rotation"
            rotation_durations = []
            for rotation in rotation_sequence:
                if rotation.get('ActualFrameDuration') is None:
                    logging.error('  ERROR IPP {}, SUID {}: missing "ActualFrameDuration" (0018, 1242) tag'
                        .format(series_row['PatientID'], series_row['SeriesInstanceUID']))
                    return
                frame_dur = int(rotation['ActualFrameDuration'].value)
                n_frames = rotation.get('NumberOfFramesInRotation')
                if n_frames is None:
                    n_frames = len(rotation.get('RadialPosition'))
                rotation_durations.append(frame_dur * int(n_frames))

            if rotation_durations is not None:
                # calculate the sum of all durations and convert it to seconds
                series_duration = sum(rotation_durations)  / 1000
                logging.debug('  {}: duration is based on rotation sequence'.format(series_row['SeriesInstanceUID']))

        # if no "phase sequence vector" is present, use the actual frame duration
        elif str(series_row['ActualFrameDuration']) != 'nan' and str(series_row['NumberOfFrames']) != 'nan':
            # calculate the duration and convert it to seconds
            series_duration = (int(series_row['ActualFrameDuration']) * series_row['NumberOfFrames']) / 1000
            logging.debug('  {}: duration is based on ActualFrameDuration'.format(series_row['SeriesInstanceUID']))

    except Exception as e:
        logging.error('Error while getting end time for IPP {}, SUID {}'
            .format(series_row['PatientID'], series_row['SeriesInstanceUID']))
        logging.error("-"*60)
        logging.error(e, exc_info=True)
        logging.error("-"*60)

    # if a duration could *not* be extracted
    if series_duration is None:
        logging.error('  ERROR IPP {}, SUID {}: no series duration found'
            .format(series_row['PatientID'], series_row['SeriesInstanceUID']))
        return

    # if a duration info could be extracted, calculate the duration from the last instance's
    #   start time, as there could be multiple instances, even for a 'NM' series
    start_time_str = str(series_row['AcquisitionTime']).split('.')[0]
    if start_time_str is None or len(start_time_str) == 0 or start_time_str == 'nan':
        return None

    start_time = dt.strptime(start_time_str, '%H%M%S')
    end_time = start_time + timedelta(seconds=series_duration)
    end_time_str = end_time.strftime('%H%M%S')

    return end_time_str

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

def get_data(config, query_dicts, to_fetch_fields, delete_data=True):
    """
    Read in as a DataFrame the data downloaded from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_dicts (list of dict): list of key-value pair dictionnaries of the parameters for each query
        to_fetch_fields (list): list of field names to fetch from the DICOM Dataset object
        delete_data (bool): whether or not to delete the DICOM files after reading them
    Returns:
        df (DataFrame): a DataFrame containing all retrieved data
    """


    # make sure download folder exists
    if not os.path.exists(config['path']['dicom_temp_dir']):
        os.makedirs(config['path']['dicom_temp_dir'])

    # make sure download folder is empty
    delete_DICOM_files(config)
    # first download the DICOM files
    download_data_dcm4che(config, query_dicts)
    # then read in the DICOM files
    df = read_DICOM_files(config, to_fetch_fields)
    # if required, delete the DICOM files
    if delete_data:
        delete_DICOM_files(config)

    return df

def download_data_dcm4che(config, query_dicts):
    """
    Download the data specified by the query dictionary from the PACS using the
        dcm4che toolkit (movescu & storescp exe files).
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_dicts (list of dict): list of key-value pair dictionnaries of the parameters for each query
    Returns:
        None
    """

    logging.info('Getting data for {} queries'.format(len(query_dicts)))

    # create the command to launch the Store SCP server receiving the DICOMs
    storescp_commands = [
        '{}/storescp.bat'.format(config['path']['dcm4che_path'])
            .replace('/', '\\').replace('\\\\', '\\'),
        '-b', '{local_ae_title}@{local_host}:{local_port}'.format(**config['PACS']),
        '--directory', '{}'.format(config['path']['dicom_temp_dir'])
            .replace('/', '\\').replace('\\\\', '\\')
    ]

    # create the command to launch the MOVE SCU command to tell the PACS to send the DICOMs
    movescu_commands = [
        '{}/movescu.bat'.format(config['path']['dcm4che_path'])
            .replace('/', '\\').replace('\\\\', '\\'),
        '-c', '{remote_ae_title}@{remote_host}:{remote_port}'.format(**config['PACS']),
        '-b', '{local_ae_title}@{local_host}:{local_port}'.format(**config['PACS']),
        '--dest', '{local_ae_title}'.format(**config['PACS']),
        '-L', 'IMAGE'
    ]

    # log the commands
    logging.debug(storescp_commands)
    logging.debug(movescu_commands)

    # catch errors because we want to make sure to kill the processes we spawn
    try:

        # launch the server listening process
        storescp_process = subprocess.Popen(storescp_commands,
                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        logging.debug('Created storescp process (PID={})'.format(storescp_process.pid))

        # store all the querying processes
        movescu_processes = []

        # for each query, send a C-MOVE query
        logging.debug('Launching {} queries'.format(len(query_dicts)))
        movescu_start_time = dt.now()
        for query_dict in query_dicts:

            # tansform the dictionnary to a list of filtering parameters
            filter_params = []
            for key, item in query_dict.items():
                filter_params.extend(['-m', '{}={}'.format(key, item)])

            # launch the query process
            logging.debug(movescu_commands + filter_params)
            movescu_process = subprocess.Popen(movescu_commands + filter_params,
                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            movescu_processes.append(movescu_process)
            logging.debug('Created movescu process (PID={})'.format(movescu_process.pid))

        # wait until the C-MOVE is done
        logging.info('Waiting for responses')
        while any([movescu_process.poll() is None for movescu_process in movescu_processes]):
            logging.debug('Waiting for C-MOVE to happen')
            time.sleep(1)
            elapsed_time = (dt.now() - movescu_start_time).seconds
            if elapsed_time > config['retrieve'].getint('movescu_timeout'):
                logging.warning('Responses timeout (elapsed: {} seconds)'.format(elapsed_time))
                break

    # catch errors
    except Exception as e:
        logging.error('Error while getting data using dcm4che: {}'.format(str(e)))

    # make sure the process is stopped
    finally:

        logging.debug('Killing processes')

        # kill all the movescu process
        for movescu_process in movescu_processes:
            logging.debug('Killing movescu process')
            movescu_process.terminate()

        # kill the storescp process
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


def read_DICOM_files(config, to_fetch_fields):
    """
    Read in as a DataFrame the data downloaded from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        to_fetch_fields (list): list of field names to fetch from the DICOM Dataset object
    Returns:
        df (DataFrame): a DataFrame containing all the read data
    """

    # get the list of all available files in folder
    DICOM_file_names = os.listdir(config['path']['dicom_temp_dir'])
    logging.warning('Reading in {} DICOM file(s)'.format(len(DICOM_file_names)))
    # create the Dataframe that will hold the content of each file
    df = pd.DataFrame(columns=to_fetch_fields)
    # go through the files
    i = 0
    for DICOM_file_name in DICOM_file_names:
        # read the DICOM file, without reading the pixels
        ds = dcmread(os.path.join(config['path']['dicom_temp_dir'], DICOM_file_name),
            stop_before_pixels=True)
        # copy each field back into the DataFrame
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

    for filename in os.listdir(config['path']['dicom_temp_dir']):
        file_path = os.path.join(config['path']['dicom_temp_dir'], filename).replace('\\', '/').replace('//', '/')
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error('Failed to delete {}. Reason: {}'.format(file_path, str(e)))
