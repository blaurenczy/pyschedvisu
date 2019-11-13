#!/usr/bin/env python

import logging
import time
import os

from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
from pandas import DataFrame

from pydicom.dataset import Dataset
from pydicom.tag import Tag

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
    
    # set debug level of pynetdicom based on the configuration file's content
    logging.getLogger('pynetdicom').setLevel(config['main']['pynetdicom_debug_level'])

    # get the date range from the config
    start_date = dt.strptime(config['main']['start_date'], '%Y-%m-%d')
    end_date = dt.strptime(config['main']['end_date'], '%Y-%m-%d')
    days_range = pd.date_range(start_date, end_date)
    
    # go through the date range day by day
    for day in days_range:
        logging.info('Processing {}'.format(day.strftime("%Y%m%d")))
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
        # exclude series where no information could be gathered
        df_series = df_series[~df_series.end_time.isnull()]
        # make sure the save directory exists
        if not os.path.exists(day_save_dir_path): os.makedirs(day_save_dir_path)
        # save the series
        df_series.to_pickle(day_save_file_path)


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

        # get the institution name(s) for this study based on the found series
        inst_names = list(set([inst_name.replace('  ', ' ') for inst_name in df_series_for_study.loc[:, 'Institution Name']]))
        # if we found multiple institution names
        if len(inst_names) > 1:
            logging.warning('Multiple institution names for study: "{}"'.format(' / '.join(inst_names)))
            inst_name = 'mixed'
        # if we found a single institution name
        else:
            inst_name = inst_names[0]
        # set the institution name for this study
        df_studies.loc[i_study, 'Institution Name'] = inst_name

        # filter for the institution name
        accepted_inst_names = config['retrieve']['accepted_institution_names'].split('\n')
        # if this instiution name is not in the list of accepted institution names, skip it
        if inst_name.lower().replace(' ', '') not in accepted_inst_names:
            logging.warning('Skipping study because it is not from CHUV (but from "{}")'.format(inst_name))
            continue

        logging.info('Appending {} series'.format(len(df_series_for_study)))
        # append the new series to the main series DataFrame
        df_series = df_series.append(df_series_for_study, sort=False, ignore_index=True)

    # add some required columns
    df_series['start_time'] = None
    df_series['end_time'] = None
    df_series['machine'] = None
            
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


def fetch_info_for_series(config, df_series, i_try_field_name='i_try'):
    """
    Get some information (start & end time, machine, etc.) for each series based on the images found in the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        df_series (DataFrame): a pandas DataFrame holding the series
        i_try_field_name (str): name of the field where to store the "number of tries" information
    Returns:
        df_series (DataFrame): a pandas DataFrame holding the series
    """

    logging.info('Going through {} series'.format(len(df_series)))
    
    # go through each series severall time (overall "pass")
    for i_overall_try in range(int(config['retrieve']['n_max_overall_try'])):
    
        # change the name of the field storing the number of tries
        i_try_field_name = 'i_try_{}'.format(i_overall_try)
        df_series[i_try_field_name] = None
        
        # go through each series
        for i_series in df_series.index:
            
            # skip series where information already exists
            if df_series.loc[i_series, 'start_time'] is not None: continue
            
            row_info, i_try = None, 0
            while row_info is None:
                i_try += 1
                df_series.loc[i_series, i_try_field_name] = i_try
                # find information about this series by fetching some images
                row_info = fetch_info_for_single_series(config, df_series.loc[i_series])
                # if there is no data and we reached our maximum number of tries
                if row_info is None and i_try >= int(config['retrieve']['n_max_try']):
                    # mark row as a failed trial and abort
                    df_series.loc[i_series, i_try_field_name] = -1
                    break
                # if there is no data but we did not reach (yet) our maximum number of tries
                elif row_info is None:
                    # delay the next retry
                    time.sleep(float(config['retrieve']['inter_try_sleep_time']))
                    
            # abort processing for this series no data
            if row_info is None:
                logging.error('ERROR with series {}: no data found'.format(df_series.loc[i_series, 'Series Instance UID']))
                continue

            # copy the relevant parameters into the main DataFrame
            df_series.loc[i_series, 'start_time'] = row_info['start_time']
            df_series.loc[i_series, 'end_time'] = row_info['end_time']
            df_series.loc[i_series, 'machine'] = row_info['machine']

    return df_series


def show_stats_for_fetching_series_info(df_series):
    """
    Show some statistics on the fetching of information for seriess.
    Args:
        df_series (DataFrame): a pandas DataFrame holding the series
    Returns:
        None
    """

    n = len(df_series)
        
    # if we have the information about the first round
    if 'i_try_0' in df_series.columns:
        # count successfull and failed trials on the first round
        i_try_0 = df_series['i_try_0']
        failures = i_try_0[i_try_0 == -1]
        n_fail = len(failures)
        successes = i_try_0[i_try_0 != -1]
        n_succ = len(successes)
        first = successes[successes == 1]
        multi = successes[successes > 1]
        # print out the stats for the first round
        logging.info('Success     (1): {:03d} / {:03d} ({:.1f}%)'.format(n_succ, n, 100 * n_succ / n))
        logging.info('Failures    (1): {:03d} / {:03d} ({:.1f}%)'.format(n_fail, n, 100 * n_fail / n))
        logging.info('First tries (1): {:03d} / {:03d} ({:.1f}%)'.format(len(first), n_succ, 100 * len(first) / n_succ))
        logging.info('Multi-tries (1): {:03d} / {:03d} ({:.1f}%)'.format(len(multi), n_succ, 100 * len(multi) / n_succ))
        logging.info('Mean ± SD multi-tries (1): {:.2f} ± {:.2f}'.format(multi.mean(), multi.std()))

        # if we have the information about the second round
        if 'i_try_1' in df_series.columns and n_fail > 0:
            # count successfull and failed trials on the second round
            i_try_1 = df_series['i_try_1']
            recoveries = i_try_1[(i_try_0 == -1) & (i_try_1 != -1)]
            n_recov = len(recoveries)
            total_failures = i_try_1[(i_try_0 == -1) & (i_try_1 == -1)]
            recov_first = recoveries[recoveries == 1]
            recov_multi = recoveries[recoveries > 1]
            # print out the stats for the second round
            logging.info('Recoveries  (2):  {:03d} / {:03d} ({:.1f}%)'.format(n_recov, n_fail, 100 * n_recov / n_fail))
            logging.info('Total fails (2):  {:03d} / {:03d} ({:.1f}%)'.format(len(total_failures), n_fail, 100 * len(total_failures) / n_fail))
            logging.info('First tries (2): {:03d} / {:03d} ({:.1f}%)'.format(len(recov_first), n_recov, 100 * len(recov_first) / n_recov))
            logging.info('Multi-tries (2): {:03d} / {:03d} ({:.1f}%)'.format(len(recov_multi), n_recov, 100 * len(recov_multi) / n_recov))
            logging.info('Mean ± SD multi-tries (2): {:.2f} ± {:.2f}'.format(recov_multi.mean(), recov_multi.std()))


def fetch_info_for_single_series(config, series_row):
    """
    Get some information (start & end time, machine, etc.) for a single series based on the images found in the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        series_row (Series): a pandas Series (row) specifying the series to query
    Returns:
        info (dict): a dictionary containing all the retrieved information
    """

    # fields to use as filters to get the image
    image_level_filters = ['Patient ID', 'Series Date', 'Series Instance UID', 'Modality']

    # fields to fetch from the DICOM header
    to_fetch_fields = ['InstanceNumber', 'ManufacturerModelName', 'AcquisitionTime', 'Modality',
                       'ImageType', 'ActualFrameDuration', 'NumberOfFrames', '0x00540032', '0x00540052']

    # create an information string for the logging of the current series
    UID = series_row['Series Instance UID']
    series_string = '[XXX]: {}|{}|{}|IPP:{:7s}|{}...{}'.format(*series_row[
        ['Series Date', 'Series Time', 'Modality', 'Patient ID']], UID[:8], UID[-4:])
    if 'i_try_0' in series_row and 'i_try_1' in series_row:
        series_string = series_string.replace('XXX', '{:3d}|{:2d}|{:2d}'.format(series_row.name, *series_row[['i_try_0', 'i_try_1']]))
    if 'i_try_0' in series_row:
        series_string = series_string.replace('XXX', '{:3d}|{:2d}'.format(series_row.name, series_row.i_try_0))
    else:
        series_string = series_string.replace('XXX', '{:3d}'.format(series_row.name))
    # actually do the logging :-) This part is probably overly complicated, but it looks nice on the logging output!
    logging.info('Fetching info for {}'.format(series_string))

    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(series_row, 'IMAGE', incl=image_level_filters)
    # add some more filters for the 'PT' modality
    if series_row['Number of Series Related Instances'] != '1':
        image_number_filter = ['1', series_row['Number of Series Related Instances']]
    else:
        image_number_filter = '1'
    # add the instance number filters for the 'PT' & 'CT' modalities
    if series_row['Modality'] == 'PT' or series_row['Modality'] == 'CT':
        query_ds.InstanceNumber = image_number_filter
    # display the Dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # fetch the data (C-MOVE)
    df, datasets = get_data(config, query_ds, to_fetch_fields)

    # sort the data and reset the index. Warning: this does not guarantee that the first index (0) has
    #    the highest InstanceNumber. Data is sorted according to the AcquisitionTime
    df = df.sort_values('AcquisitionTime')
    df.reset_index(drop=True, inplace=True)

    # if no data is found, skip with error
    if len(df) == 0:
        logging.debug('  ERROR for {}: no data found. "Series Description" field = "{}"'
                      .format(series_string, series_row['Series Description']))
        return

    # if there are too many rows, skip with error
    elif len(df) > 2:
        logging.error('  ERROR for {}: too many rows found ({})'.format(series_string, len(df)))
        return

    # if there is only one row, duplicate it
    elif len(df) == 1:
        df.loc[1, :] = df.loc[0, :]

    # if the image type is "SECONDARY", this means that we are not dealing with a raw image but with a processed image
    if 'SECONDARY' in df.loc[0, 'ImageType']:

        # try to rescue this series by looking at the before-last image
        if int(series_row['Number of Series Related Instances']) > 1:
            logging.warning('  WARNING for {}: secondary image type found. Trying to recover'.format(series_string))
            # try to do another query for the before-last image
            query_ds.InstanceNumber = str(int(series_row['Number of Series Related Instances']) - 1)
            # fetch the data (C-MOVE)
            df_before_last, _ = get_data(config, query_ds, to_fetch_fields)
            # if no data was found, abort
            if len(df_before_last) == 0:
                logging.error('  ERROR for {}: secondary image type found: "{}". "Series Description" = "{}"'
                          .format(series_string, '-'.join(df.loc[0, 'ImageType']), series_row['Series Description']))
                return
            # if data is still secondary, abort
            elif 'SECONDARY' in df_before_last.iloc[0]['ImageType']: 
                logging.error('  ERROR for {}: secondary image type found: "{}". "Series Description" = "{}"'
                          .format(series_string, '-'.join(df.loc[0, 'ImageType']), series_row['Series Description']))
                return 
            # copy the data instead of the last frame
            df.loc[0, :] = df_before_last.reset_index(drop=True).loc[0, :]
            logging.info('  INFO for {}: secondary image type recovered: "{}". '
                          .format(series_string, '-'.join(df.loc[0, 'ImageType'])))

        # if this series does not have the option of going for the before-last InstanceNumber, abort
        else:
            logging.error('  ERROR for {}: secondary image type found: "{}". "Series Description" field = "{}"'
                      .format(series_string, '-'.join(df.loc[0, 'ImageType']), series_row['Series Description']))
            return

    # exrtact the start and end times
    start_time_str = str(df.loc[0, 'AcquisitionTime']).split('.')[0]
    end_time_str = str(df.loc[1, 'AcquisitionTime']).split('.')[0]

    # for modality type 'NM', the end time should be calculated
    if df.loc[0, 'Modality'] == 'NM':

        # try to get a duration for the current series
        series_duration = None

        # try to extract the "Phase Information Sequence"
        phase_sequence = df.loc[0, '0x00540032']
        # try to extract the "Rotation Information Sequence"
        rotation_sequence = df.loc[0, '0x00540052']

        if str(phase_sequence) != 'nan':
            # extract the duration of each "phase"
            phase_durations = []
            for phase in phase_sequence:
                frame_dur = int(phase['ActualFrameDuration'].value)
                n_frames = int(phase['NumberOfFramesInPhase'].value)
                phase_durations.append(frame_dur * n_frames)
            # calculate the sum of all durations and convert it to seconds 
            series_duration = sum(phase_durations)  / 1000
            logging.debug('  {}: duration is based on phase sequence'.format(series_string))

        elif str(rotation_sequence) != 'nan':
            # extract the duration of each "rotation"
            rotation_durations = []
            for rotation in rotation_sequence:
                frame_dur = int(rotation['ActualFrameDuration'].value)
                n_frames = int(rotation['NumberOfFramesInRotation'].value)
                rotation_durations.append(frame_dur * n_frames)
            # calculate the sum of all durations and convert it to seconds 
            series_duration = sum(rotation_durations)  / 1000
            logging.debug('  {}: duration is based on rotation sequence'.format(series_string))

        # if no "phase sequence vector" is present, use the actual frame duration
        elif str(df.loc[0, 'ActualFrameDuration']) != 'nan' and str(df.loc[0, 'NumberOfFrames']) != 'nan':
            # calculate the duration and convert it to seconds 
            series_duration = (int(df.loc[0, 'ActualFrameDuration']) * df.loc[0, 'NumberOfFrames']) / 1000
            logging.debug('  {}: duration is based on ActualFrameDuration'.format(series_string))

        # if a duration could *not* be extracted
        if series_duration is None:
            logging.error('  ERROR for {}: no series duration found'.format(series_string))
            return

        # if a duration could be extracted
        else:
            # calculate the duration from the last instance's start time, as there could
            #    be multiple instances, even for a 'NM' series
            start_time_str = str(df.loc[1, 'AcquisitionTime']).split('.')[0]
            start_time = dt.strptime(start_time_str, '%H%M%S')
            end_time = start_time + timedelta(seconds=series_duration)
            end_time_str = end_time.strftime('%H%M%S')

    # create a dictionary to return the gathered information
    info = {
        'start_time': start_time_str,
        'end_time': end_time_str,
        'machine': df.loc[0, 'ManufacturerModelName']
    }

    return info


def create_dataset_from_dataframe_row(df_row, qlevel, incl=[], excl=[]):
    """
    Creates a pydicom Dataset for querying based on the content of an input DataFrame's row, provided
    as a Series.
    Args:
        df_row (Series): a DataFrame row containing all the information to create the query
        qrlevel (str): a string specifying the query (retrieve) level (PATIENT / STUDY / SERIES / IMAGE)
        incl (list): list of the columns to include in the Dataset. By default: all.
        excl (list): list of the columns to exlude in the Dataset. By default: none.
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

    return ds


def _handle_result(event):
    """ Handle the result of the query request """
    logging.debug('Found a Dataset: {}'.format(event.dataset.SOPInstanceUID))
    # get the next index
    i = len(df) + 1
    datasets.append(event.dataset)
    # copy all requested information
    for col in df.columns:
        logging.debug("Setting col '{}' for index {}".format(col, i))
        if Tag(col) in event.dataset.keys():
            df.loc[i, col] = event.dataset[col].value
            logging.debug("Setting col '{}' for index {} with value '{}'".format(col, i, event.dataset[col].value))
    # return a "Success" code
    return 0x0000


def get_data(config, query_dataset, to_fetch_fields):
    """
    Retrieve the data specified by the query dataset from the PACS using the C-MOVE mode.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_dataset (pydicom.dataset.Dataset): a Dataset object holding the filtering parameters
        to_fetch_fields (list): a list of strings specifying the fields to retrieve
    Returns:
        df (DataFrame): a DataFrame containing all retrieved data
        datasets (list): a list of the retrieved datasets
    """

    logging.debug("Getting data")

    # initialize a global DataFrame to store the results
    global df
    df = DataFrame(columns = to_fetch_fields)
    # initialize a global list of Datasets to store the results
    global datasets
    datasets = []
    # define the handlers
    handlers = [(evt.EVT_C_STORE, _handle_result)]
    # initialise the Application Entity
    ae = AE(ae_title = config['PACS']['ae_title'])
    # add a requested presentation context
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)
    # add the Storage SCP's supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    # start our Storage SCP in non-blocking mode
    logging.debug("Creating receiving server")
    scp = ae.start_server((config['PACS']['local_host'], config['PACS'].getint('local_port')),
                          block=False, evt_handlers=handlers)
    logging.debug("Connecting to PACS")
    # Associate with peer AE at IP 127.0.0.1 and port 11112
    assoc = ae.associate(config['PACS']['host'], config['PACS'].getint('port'),
                         ae_title=config['PACS']['ae_called_title']) 
    try: 
        # if the connection is successfully established
        if assoc.is_established:
            logging.debug("Association established")
            # use the C-MOVE service to send the identifier
            responses = assoc.send_c_move(query_dataset, config['PACS']['ae_title'],
                                          PatientRootQueryRetrieveInformationModelMove)
            logging.debug("Response(s) received")
            i_response = 0
            for (status, identifier) in responses:
                if status:
                    logging.debug('C-MOVE query status: 0x{0:04x}'.format(status.Status))
                else:
                    logging.error('Connection timed out, was aborted or received invalid response')
                logging.debug('Status: {}, identifier: {}'.format(str(status), str(identifier)))
        else:
            logging.error('Association rejected, aborted or never connected')
    except:
        logging.error('Error during fetching of data (C-MOVE)')
        raise
    finally:
        # release the association and stop our Storage SCP
        assoc.release()
        scp.shutdown()
        logging.debug("Connection closed")
    return df, datasets


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
    ae = AE(ae_title=config['PACS']['ae_title'])
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    # associate with peer AE
    logging.debug("Connecting to PACS")
    assoc = ae.associate(config['PACS']['host'], config['PACS'].getint('port'),
                         ae_title=config['PACS']['ae_called_title'])
    try:
        # if the connection is successfully established
        if assoc.is_established:
            logging.debug("Association established")
            # use the C-FIND service to send the identifier
            responses = assoc.send_c_find(query_dataset,
                                          PatientRootQueryRetrieveInformationModelFind)
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
