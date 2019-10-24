#!/usr/bin/env python

import logging

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
    Finds all studies for a single day from the PACS.
    Args:
        config (dict): a dictionary holding all the necessary parameters
        study_row (Series): a pandas Series (row) specifying the study to query
    Returns:
        df (DataFrame): a DataFrame containing all retrieved studies
    """

    logging.info("Retrieving all studies from PACS for day {}".format(study_date))
    series_level_filters = ['Study Date', 'Patient ID']
    to_drop_columns = ['Query/Retrieve Level', 'Retrieve AE Title', 'Type of Patient ID', 'Issuer of Patient ID']
    sort_columns = ['Series Time', 'Number of Series Related Instances']

    logging.info('DataFrame row:\n' + str(study_row))

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
    logging.info('Query Dataset:')
    for s in str(query_ds).split('\n'): logging.info('    ' + s)

    # do the query (C-FIND)
    df_series = find_data(config, query_ds)

    # drop unwanted columns, sort and display
    df_series = df_series.drop(to_drop_columns, axis=1)
    df_series.sort_values(sort_columns, inplace=True)
    df_series.reset_index(drop=True, inplace=True)

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
