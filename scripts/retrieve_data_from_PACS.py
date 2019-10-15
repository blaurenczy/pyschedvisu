#!/usr/bin/env python

import logging

from pandas import DataFrame

from pydicom.dataset import Dataset

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

def _handle_result(event):
        """ Handle the result of the query request """
        logging.info('Found a Dataset: {}'.format(event.dataset.SOPInstanceUID))
        # get the next index
        i = len(df) + 1
        # copy all requested information
        for col in df.columns:
            df.loc[i, col] = event.dataset.get(col)
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
    """
    
    # initialize a global DataFrame to store the results
    global df
    df = DataFrame(columns = return_fields)
    # define the handlers
    handlers = [(evt.EVT_C_STORE, _handle_result)]
    # initialise the Application Entity
    ae = AE(ae_title = config['PACS']['ae_title'])
    # add a requested presentation context
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)
    # add the Storage SCP's supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    # start our Storage SCP in non-blocking mode
    logging.debug("Creatin receiving server")
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
    return df

    
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
