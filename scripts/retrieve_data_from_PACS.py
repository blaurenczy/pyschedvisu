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
    
    
def retrieve_data(config, query_dataset):
    """
    Retrieve the data specified by the config and the query dataset from the PACS.
    Args:
        config (dict):      a dictionary holding all the necessary parameters
    Returns:
        df (DataFrame):     a DataFrame containing all retrieved data
    """
    
    logging.debug("Connecting to PACS")
    
    # create the AE with the "find" information model
    ae = AE(ae_title=config['PACS']['ae_title'])
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)

    # associate with peer AE
    assoc = ae.associate(config['PACS']['host'], config['PACS'].getint('port'), ae_title=config['PACS']['ae_called_title'])
    
    # initialize a DataFrame to store the results
    df = DataFrame()

    # if the connection is made
    if assoc.is_established:
        
        logging.debug("Association established")
        
        # use the C-FIND service to send the identifier
        responses = assoc.send_c_find(query_dataset, PatientRootQueryRetrieveInformationModelFind)
        
        logging.debug("Response received")
        i_response = 0
        for (status, identifier) in responses:
            if status:
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
