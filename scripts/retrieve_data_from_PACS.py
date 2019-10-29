#!/usr/bin/env python

import logging

from datetime import datetime as dt
from datetime import timedelta

from pandas import DataFrame

from IPython.core import display as ICD

from pydicom.dataset import Dataset
from pydicom.tag import Tag

from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelMove
)


def retrieve_data_from_PACS(config):
    """
    Retrieve the relevant (meta-)data from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
    Returns:
        None
    """

    logging.info("Retrieving data from PACS")

    return


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

    logging.info("Retrieving all studies from PACS for day {}".format(study_date))

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
    query_ds.ReferringPhysicianName = ''
    query_ds.InstitutionName = ''

    # display the query dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # do the query (C-FIND)
    df_studies = find_data(config, query_ds)

    # drop unwanted columns and display
    to_drop_columns = ['Query/Retrieve Level', 'Retrieve AE Title', 'Type of Patient ID',
                        'Issuer of Patient ID']
    df_studies = df_studies.drop(to_drop_columns, axis=1)

    return df_studies


def find_series_for_study(config, study_row):
    """
    Finds all series for a study from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        study_row (Series): a pandas Series (row) specifying the study to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved series
    """

    logging.info("Retrieving all series for study [{}]: {}".format(study_row.name, study_row['Study Instance UID']))

    series_level_filters = ['Study Date', 'Patient ID']
    to_drop_columns = ['Query/Retrieve Level', 'Retrieve AE Title', 'Type of Patient ID', 'Issuer of Patient ID']
    sort_columns = ['Series Time', 'Number of Series Related Instances']

    logging.debug('DataFrame row:\n' + str(study_row))

    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(study_row, 'SERIES', incl=series_level_filters)

    # parameters for filtering
    query_ds.SeriesDate = query_ds.StudyDate
    query_ds.Modality = ['NM', 'PT', 'CT']

    # parameters to fetch
    query_ds.SeriesInstanceUID = ''
    query_ds.StudyInstanceUID = ''
    query_ds.SeriesTime = ''
    query_ds.StudyTime  = ''
    query_ds.NumberOfSeriesRelatedInstances = ''
    query_ds.SeriesDescription = ''
    query_ds.SeriesNumber = ''
    query_ds.ProtocolName = ''
    query_ds.InstitutionName = ''
    query_ds.RefferingPhysicianName = ''

    # display the query dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # do the query (C-FIND)
    df_series = find_data(config, query_ds)

    # filter out some Series that are not primary acquisitions (and do not contain any relevant time information)
    series_descr_patterns_to_exclude = ['.+statistics$', '.+report$', '.+results$', '.+protocol$', '^defaultseries$',
        '^results.+', '^fusion.*', '^processed images.*', '^4DM.+', '.+SUV5$', '^save_screens$', '^key_images$',
        '^fused.+', '^mip.*', '^mpr\..*', '^compact.+', '^images medrad intego$']
    for descr_pattern in series_descr_patterns_to_exclude:
        to_exclude = df_series['Series Description'].str.match(descr_pattern, case=False)
        if to_exclude.sum() > 0:
            logging.info('Found {} series to exclude based on their description: "{}"'.format(to_exclude.sum(),
                '", "'.join(df_series[to_exclude]['Series Description'])))
        df_series = df_series[~to_exclude]
    
    series_protocol_to_exclude = ['SCREENCAPTURE']
    df_series = df_series[~df_series['Protocol Name'].isin(series_protocol_to_exclude)]

    # drop unwanted columns, sort and display
    df_series = df_series.drop(to_drop_columns, axis=1)
    df_series.sort_values(sort_columns, inplace=True)
    df_series.reset_index(drop=True, inplace=True)

    return df_series


def fetch_info_for_series(config, series_row):
    """
    Get some information (start & end time, machine, etc.) for a series based on the images found in the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        series_row (Series): a pandas Series (row) specifying the series to query
    Returns:
        info (dict): a dictionary containing all the retrieved information
    """ 
    
    # fields to use as filters to get the image
    image_level_filters = ['Patient ID', 'Study Date', 'Series Instance UID', 'Modality']
    
    # fields to fetch from the DICOM header
    to_fetch_fields = ['InstanceNumber', 'ManufacturerModelName', 'AcquisitionTime', 'Modality',
                       'ImageType', 'ActualFrameDuration', 'NumberOfFrames', '0x00540032', '0x00540052']
    
    # create an information string for the logging of the current series
    UID = series_row['Series Instance UID']
    series_string = '[{}]: {}|{}|{}|PID:{}|{}...{}'.format(series_row.name, *series_row[
        ['Study Date', 'Series Time', 'Modality', 'Patient ID']], UID[:8], UID[-4:])
    logging.info('Fetching info for {}'.format(series_string))
    
    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(series_row, 'IMAGE', incl=image_level_filters)
    # add some more filters for the 'PT' modality
    if series_row['Modality'] == 'PT' or series_row['Modality'] == 'CT' or series_row['Number of Series Related Instances'] != '1':
        query_ds.InstanceNumber = ['1', series_row['Number of Series Related Instances']]
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
        logging.error('  ERROR for {}: no data found. "Series Description" field = "{}"'
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
        #'ds': datasets[0]
    }
    
    return info


def prunes_series_by_time_overlap(df):
    """
    Prune the input series DataFrame based on start/end time overlaps.
    Args:
        df (DatFrame): a pandas DataFrame to check for time overlaps
    Returns:
        df (DatFrame): the pruned pandas DataFrame
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
        # release the association
        assoc.release()
    else:
        logging.error('Association rejected, aborted or never connected')
    # stop our Storage SCP
    scp.shutdown() 
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

         # release the association
        assoc.release()
        logging.debug("Connection closed")
    else:
        logging.error('Association rejected, aborted or never connected')
        
    return df
