"""
Provide functions for reading and parsing configuration files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config

from .staticargs import amend_static_args, typecast_static_args


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def read_ini_config(file_path):
    """
    Read in a `*.ini` config file, which is used mostly to specify the
    waveform simulation (for example, the waveform model, the parameter
    space for the binary black holes, etc.) and return its contents.
    
    Args:
        file_path (str): Path to the `*.ini` config file to be read in.

    Returns:
        A tuple `(variable_arguments, static_arguments)` where
        
        * `variable_arguments` should simply be a list of all the
          parameters which get randomly sampled from the specified
          distributions, usually using an instance of
          :class:`utils.waveforms.WaveformParameterGenerator`.
        * `static_arguments` should be a dictionary containing the keys
          and values of the parameters that are the same for each
          example that is generated (i.e., the non-physical parameters
          such as the waveform model and the sampling rate).
    """
    
    # Make sure the config file actually exists
    if not os.path.exists(file_path):
        raise IOError('Specified configuration file does not exist: '
                      '{}'.format(file_path))
    
    # Set up a parser for the PyCBC config file
    workflow_config_parser = WorkflowConfigParser(configFiles=[file_path])
    
    # Read the variable_arguments and static_arguments using the parser
    variable_arguments, static_arguments = \
        read_params_from_config(workflow_config_parser)
    
    # Typecast and amend the static arguments
    static_arguments = typecast_static_args(static_arguments)
    static_arguments = amend_static_args(static_arguments)
    
    return variable_arguments, static_arguments


def read_json_config(file_path):
    """
    Read in a `*.json` config file, which is used to specify the
    sample generation process itself (for example, the number of
    samples to generate, the number of concurrent processes to use,
    etc.) and return its contents.
    
    Args:
        file_path (str): Path to the `*.json` config file to be read in.

    Returns:
        A `dict` containing the contents of the given JSON file.
    """
    
    # Make sure the config file actually exists
    if not os.path.exists(file_path):
        raise IOError('Specified configuration file does not exist: '
                      '{}'.format(file_path))
    
    # Open the config while and load the JSON contents as a dict
    with open(file_path, 'r') as json_file:
        config = json.load(json_file)

    # Define the required keys for the config file in a set
    required_keys = {'background_data_directory', 'dq_bits', 'inj_bits',
                     'waveform_params_file_name', 'max_runtime',
                     'n_injection_samples', 'n_noise_samples', 'n_processes',
                     'random_seed', 'output_file_name'}
    
    # Make sure no required keys are missing
    missing_keys = required_keys.difference(set(config.keys()))
    if missing_keys:
        raise KeyError('Missing required key(s) in JSON configuration file: '
                       '{}'.format(', '.join(list(missing_keys))))

    return config
