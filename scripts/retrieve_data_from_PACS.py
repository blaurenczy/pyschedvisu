#!/usr/bin/env python

import logging

from datetime import datetime as dt

from pandas import DataFrame

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


def find_PT_studies_for_day(config, study_date):
    """
    Finds all PT studies for a single day from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        study_date (str): a string specifying the day to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved studies
    """

    logging.info("Retrieving all studies from PACS for day {}".format(study_date))

    # create the query dataset
    query_ds = Dataset()

    # parameters for filtering
    query_ds.QueryRetrieveLevel = 'STUDY'
    query_ds.ModalitiesInStudy = 'PT'
    query_ds.StudyDate = study_date

    # parameters to fetch
    query_ds.StudyTime  = ''
    query_ds.StudyInstanceUID = ''
    query_ds.PatientID = ''
    query_ds.StudyDescription = ''
    # query_ds.InstitutionName = ''
    # query_ds.ReferringPhysicianName = ''

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

    logging.info("Retrieving all series from PACS for study {}".format(study_row['Study Instance UID']))

    series_level_filters = ['Study Date', 'Patient ID']
    to_drop_columns = ['Query/Retrieve Level', 'Retrieve AE Title', 'Type of Patient ID', 'Issuer of Patient ID']
    sort_columns = ['Series Time', 'Number of Series Related Instances']

    logging.debug('DataFrame row:\n' + str(study_row))

    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(study_row, 'SERIES', incl=series_level_filters)

    # parameters for filtering
    query_ds.SeriesDate = query_ds.StudyDate
    query_ds.Modality = ['PT', 'CT']

    # parameters to fetch
    query_ds.SeriesInstanceUID = ''
    query_ds.StudyInstanceUID = ''
    query_ds.SeriesTime = ''
    query_ds.StudyTime  = ''
    query_ds.NumberOfSeriesRelatedInstances = ''
    query_ds.SeriesDescription = ''

    # display the query dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # do the query (C-FIND)
    df_series = find_data(config, query_ds)

    # filter out some Series that do not contain any information (no AcquisitionTime or such)
    series_descr_to_exclude = ['PET Statistics', 'Patient Protocol', 'PET Dose Report', 'Results MM Oncology Reading']
    df_series = df_series[~df_series['Series Description'].isin(series_descr_to_exclude)].reset_index(drop=True)

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
    
    image_level_filters = ['Patient ID', 'Study Date', 'Series Instance UID', 'Modality']
    to_fetch_params = ['ManufacturerModelName', 'AcquisitionTime']

    logging.info('Fetching info for {}|{}|{}|PID:{}|{}...{}'.format(*series_row[['Study Date', 'Series Time', 'Modality',
        'Patient ID']], series_row['Series Instance UID'][:8], series_row['Series Instance UID'][-4:]))
    
    # create the query dataset
    query_ds = create_dataset_from_dataframe_row(series_row, 'IMAGE', incl=image_level_filters)
    # add some more filters
    query_ds.InstanceNumber = ['1', series_row['Number of Series Related Instances']]
    # display the Dataset
    logging.debug('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.debug('    ' + s)

    # fetch the data (C-MOVE)
    dfs, _ = get_data(config, query_ds, to_fetch_params)
    dfs.reset_index(drop=True, inplace=True)
    
    # if no data is found, skip with error
    if len(dfs) == 0:
        logging.error('  ERROR, no data found.')
        return None
    # if there is only one row, duplicate it
    elif len(dfs) == 1:
        dfs.loc[1, :] = dfs.loc[0, :]
    # if there are too many rows, skip with error
    elif len(dfs) > 2:
        logging.error('  ERROR, too many rows found ({}).'.format(len(df)))
        return None
    
    # create a dictionary to return the gathered information
    info = {
        'start_time': str(df.loc[0, 'AcquisitionTime']).split('.')[0],
        'end_time': str(df.loc[1, 'AcquisitionTime']).split('.')[0],
        'machine': df.loc[0, 'ManufacturerModelName']
    }
    
    return info



def prunes_series_by_time_overlap(df_series):
    """
    Prune the input series DataFrame based on start/end time overlaps.
    Args:
        df_series (DatFrame): a pandas DataFrame to check for time overlaps
    Returns:
        df_series (DatFrame): the pruned pandas DataFrame
    """
    
    logging.info("Pruning series for time overlap")
    
    # time format
    FMT = '%H%M%S'
    
    # remove duplicates and sort (rows that have exactly the same start/end times are redundant)
    df_series.drop_duplicates(['start_time', 'end_time'], inplace=True)
    df_series = df_series[(~df_series['start_time'].isnull()) & (df_series['start_time'] != 'nan')]
    df_series = df_series.sort_values('start_time')

    # prune series based on start/end time overlaps:
    #   as long as some overlap was found, start over
    overlap_found = True
    while overlap_found:

        # make sure we only loop if an overlap was found, and reset the index of the DataFrame
        overlap_found = False
        df_series.reset_index(drop=True, inplace=True)

        # go through each row
        for i_serie in range(1, len(df_series)):

            # get the start/end times of the current row
            curr_start = dt.strptime(df_series.iloc[i_serie]['start_time'], FMT)
            curr_end = dt.strptime(df_series.iloc[i_serie]['end_time'], FMT)
            # cget the start/end times of the previous row
            prev_start = dt.strptime(df_series.iloc[i_serie - 1]['start_time'], FMT)
            prev_end = dt.strptime(df_series.iloc[i_serie - 1]['end_time'], FMT)

            # check for an overlap between the current and the previous row
            latest_start = max(curr_start, prev_start)
            earliest_end = min(curr_end, prev_end)
            delta = (earliest_end - latest_start).seconds
            overlap = max(0, delta)

            # if there is no overlap, then the current time range is fully
            #   overlapping with the previous row's range
            overlap_found = overlap == 0
            logging.debug('{:2}/{}: checking if {}-{} is included in {}-{}: overlap = {}'.format(i_serie, len(df_series) - 1,
                curr_start.strftime(FMT), curr_end.strftime(FMT), prev_start.strftime(FMT), prev_end.strftime(FMT), overlap_found))

            # if any overlap was found, remove the redundant row and start over
            if overlap_found:
                df_series.drop(i_serie,inplace=True)
                break

    return df_series
   
   
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


def get_data(config, query_dataset, return_fields):
    """
    Retrieve the data specified by the query dataset from the PACS using the C-MOVE mode.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        query_dataset (pydicom.dataset.Dataset): a Dataset object holding the filtering parameters
        return_fields (list): a list of strings specifying the fields to retrieve
    Returns:
        df (DataFrame): a DataFrame containing all retrieved data
        datasets (list): a list of the retrieved datasets
    """

    logging.debug("Getting data")
    
    # initialize a global DataFrame to store the results
    global df
    df = DataFrame(columns = return_fields)
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
