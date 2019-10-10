#!/usr/bin/env python

import logging
from pandas import DataFrame
from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind


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

    
def retrieve_images(config):
    """
    Retrieve the images specified by the config and the query dataset from the PACS.
    Args:
        config (dict):          a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame):         a DataFrame containing all the retrieved images
    """

    # create our query dataset
    query_dataset = Dataset()
    
    # initialize all the requested fields as empty strings
    request_fields = ['PatientID', 'StudyID', 'StudyInstanceUID', 'StudyDate', 'StudyTime', 'QueryRetrieveLevel',
        'SeriesInstanceUID', 'Modality', 'InstanceNumber']
    for field in request_fields:
        setattr(query_dataset, field, '')
      
    query_dataset.add_new([0x0008,0x0032], 'TM', '')
    query_dataset.add_new([0x0008,0x0033], 'TM', '')
    query_dataset.add_new([0x0008,0x0008], 'CS', '')
    query_dataset.add_new([0x0020,0x0012], 'IS', '')
    query_dataset.add_new([0x0020,0x0013], 'IS', '')
    query_dataset.add_new([0x0008,0x002A], 'DT', '')
    query_dataset.add_new([0x0020,0x1002], 'IS', '')
    query_dataset.add_new([0x0008,0x1090], 'LO', '')
    
    print(query_dataset)
    
    # set the filtering parameters
    query_dataset.StudyDate = '{}-{}'.format(config['main']['start_date'], config['main']['end_date'])
    query_dataset.QueryRetrieveLevel = 'IMAGE'
    query_dataset.StudyTime = '080000-090000'
    query_dataset.Modality = ['PT', 'CT']
    #query_dataset.InstanceNumber = 1
    #query_dataset.SeriesInstanceUID = "1.2.840.113619.2.452.3.1771786559.266.1569901094.215"
    query_dataset.SeriesInstanceUID = "1.3.12.2.1107.5.6.1.2013.31330119100111184984100000017"
    
    
    df = retrieve_data(config, query_dataset, request_fields)
    
    return df

    
def retrieve_series(config):
    """
    Retrieve the series specified by the config and the query dataset from the PACS.
    Args:
        config (dict):          a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame):         a DataFrame containing all the retrieved series
    """

    # create our query dataset
    query_dataset = Dataset()
    
    # initialize all the requested fields as empty strings
    request_fields = ['PatientID', 'StudyID', 'StudyInstanceUID', 'StudyDate', 'StudyTime', 'QueryRetrieveLevel',
        'SeriesInstanceUID', 'Modality', 'SeriesNumber', 'SeriesTime', 'NumberOfSeriesRelatedInstances']
    for field in request_fields:
        setattr(query_dataset, field, '')
        
    # set the filtering parameters
    query_dataset.StudyDate = '{}-{}'.format(config['main']['start_date'], config['main']['end_date'])
    query_dataset.QueryRetrieveLevel = 'SERIES'
    query_dataset.StudyTime = '080000-090000'
    query_dataset.Modality = ['PT', 'CT']
    
    df = retrieve_data(config, query_dataset, request_fields)
    
    return df

    
def retrieve_studies(config):
    """
    Retrieve the studies specified by the config and the query dataset from the PACS.
    Args:
        config (dict):          a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame):         a DataFrame containing all the retrieved studies
    """

    # create our query dataset
    query_dataset = Dataset()
    
    # initialize all the requested fields as empty strings
    request_fields = ['PatientID', 'StudyID', 'StudyInstanceUID', 'StudyDate', 'StudyTime', 'QueryRetrieveLevel']
    for field in request_fields:
        setattr(query_dataset, field, '')
        
    # set the filtering parameters
    query_dataset.StudyDate = '{}-{}'.format(config['main']['start_date'], config['main']['end_date'])
    query_dataset.QueryRetrieveLevel = 'STUDY'
    query_dataset.StudyTime = '080000-090000'
    
    df = retrieve_data(config, query_dataset, request_fields)
    
    return df
    
    
def retrieve_data(config, query_dataset, request_fields):
    """
    Retrieve the data specified by the config and the query dataset from the PACS.
    Args:
        config (dict):      a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame):     a DataFrame containing all retrieved data
    """
    
    logging.info("Connecting to PACS")
    
    # create the AE with the "find" information model
    ae = AE(ae_title=config['PACS']['ae_title'])
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)

    # associate with peer AE
    assoc = ae.associate(config['PACS']['host'], config['PACS'].getint('port'), ae_title=config['PACS']['ae_called_title'])
    
    # initialize a DataFrame to store the results
    df = DataFrame()

    # if the connection is made
    if assoc.is_established:
        # use the C-FIND service to send the identifier
        responses = assoc.send_c_find(query_dataset, PatientRootQueryRetrieveInformationModelFind)
        i_response = 0
        for (status, identifier) in responses:
            if status:
                # if the status is 'Pending' then identifier is the C-FIND response
                if status.Status in (0xFF00, 0xFF01):
                    # copy all requested fields
                    #for field in request_fields:
                    #   df.loc[i_response, field] = identifier.get(field)
                    # copy all fields
                    for data_element in identifier:
                        try:
                            if not isinstance(data_element.value, str) and (data_element.value) > 1:
                                df.loc[i_response, data_element.name] = ', '.join(data_element.value)
                            else:
                                df.loc[i_response, data_element.name] = data_element.value
                        except:
                             df.loc[i_response, data_element.name] = str(data_element.value)
                    i_response += 1
            else:
                logging.error('Connection timed out, was aborted or received invalid response')

         # release the association
        assoc.release()
    else:
        logging.error('Association rejected, aborted or never connected')
        
    return df
