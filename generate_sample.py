"""
The "main script" of this repository: Read in a configuration file and
generate synthetic GW data according to the provided specifications.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import json
import numpy as np
import os
import sys
import time

from itertools import count
from multiprocessing import Process, JoinableQueue
from tqdm import tqdm

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config

# Here we need to add the parent directory to the $PYTHONPATH, because
# apparently Python does not have a less hacky way of importing from a
# sibling directory if you are "just" in a script and not in a package
sys.path.insert(0, os.path.realpath('..'))

# Now we can even import from utils without PyCharm complaining!
from utils.HDFTools import NoiseTimeline  # noqa
from utils.WaveformTools import WaveformParameterGenerator, \
    generate_sample, amend_static_arguments  # noqa
from utils.samplefiles import SampleFile  # noqa
from utils.TypecastingTools import typecast_static_args  # noqa


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def queue_worker(arguments,
                 results_queue,
                 generate_sample):
    """
    This function will be passed to a queue worker and is responsible
    for getting a set of arguments from the arguments queue, generating
    the corresponding sample, adding it to the results queue, and
    updating the progress bar.

    Args:
        arguments: dict
            Dictionary containing the arguments for generate_sample().
        results_queue: JoinableQueue
            The queue to which the result of this worker is passed.
        generate_sample: function
            A function that can generate samples. Usually, this is:
                generate_sample(static_arguments,
                                event_time,
                                waveform_params)
            as defined in WaveformTools.py
    """
    
    # Try to generate a sample using the given arguments
    try:
        result = generate_sample(**arguments)
        results_queue.put(result)
        return True
    
    # For some arguments, LALSuite crashes during the sample generation.
    # In this case we can try again with different waveform parameters:
    except RuntimeError:
        sys.exit('Runtime Error')


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Disable output buffering ('flush' option is not available for Python 2)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # Start the stopwatch
    script_start = time.time()
    print('')
    print('GENERATE A GW DATA SAMPLE FILE')
    print('')

    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')

    # Add arguments (and set default values where applicable)
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    command_line_arguments = vars(parser.parse_args())
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in generator configuration file
    # -------------------------------------------------------------------------

    # Build the full path to the config file and make sure it exists
    config_file_name = command_line_arguments['config_file']
    config_file_path = os.path.join('..', 'config_files', config_file_name)
    if not os.path.exists(config_file_path):
        raise IOError('Specified configuration file does not exist!')

    # Read the configuration into a dict
    print('Reading and validating in configuration file...', end=' ')
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Make sure all required keys are present
    for key in ('background_data_directory', 'dq_bits', 'inj_bits',
                'waveform_params_file_name', 'n_injection_samples',
                'n_noise_samples', 'n_processes', 'random_seed',
                'output_file_name'):
        if key not in config.keys():
            raise KeyError('Missing key in configuration file: {}'.format(key))
    print('Done!')
    print()

    # -------------------------------------------------------------------------
    # Shortcuts and random seed
    # -------------------------------------------------------------------------

    # Define some useful shortcuts
    random_seed = config['random_seed']
    background_data_directory = config['background_data_directory']

    # Set the random seed for this script
    np.random.seed(config['random_seed'])

    # -------------------------------------------------------------------------
    # Ensure the waveform parameters file exists
    # -------------------------------------------------------------------------

    # Construct the path to the waveform params file
    waveform_params_file_name = config['waveform_params_file_name']
    waveform_params_file_path = \
        os.path.join('..', 'config_files', waveform_params_file_name)

    # Ensure it exists
    if not os.path.exists(waveform_params_file_path):
        raise IOError('Specified waveform parameter file does not exist!')

    # -------------------------------------------------------------------------
    # Construct a generator for sampling valid noise times
    # -------------------------------------------------------------------------

    # If the 'background_data_directory' is None, we will use synthetic noise.
    # In this case, there are no noise times (always return None).
    if config['background_data_directory'] is None:

        print('Using synthetic noise! (background_data_directory = None)\n')

        # Create a iterator that returns a fake "event time", which we will
        # use as a seed for the RNG to ensure the reproducibility of the
        # generated synthetic noise.
        # For the HDF file path that contains that time, we always return
        # None, so that we know that we need to generate synthetic noise.
        noise_times = ((1000000000 + _, None) for _ in count())

    # Otherwise, we set up a timeline object for the background noise, that
    # is, we read in all HDF files in the raw_data_directory and figure out
    # which parts of it are useable (i.e., have the right data quality and
    # injection bits set as specified in the config file).
    else:

        print('Using real noise from recordings! (background_data_directory = '
              '{})'.format(background_data_directory))
        print('Reading in raw data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = \
            NoiseTimeline(background_data_directory=background_data_directory,
                          random_seed=random_seed)

        # Create a noise time generator so that can sample valid noise times
        # simply by calling next(noise_time_generator)
        noise_times = (noise_timeline.sample(delta_t=config['delta_t'],
                                             dq_bits=config['dq_bits'],
                                             inj_bits=config['inj_bits'],
                                             return_paths=True)
                       for _ in iter(int, 1))
        print('Done!\n')

    # -------------------------------------------------------------------------
    # Construct a generator for sampling waveform parameters
    # -------------------------------------------------------------------------

    # Initialize a waveform parameter generator that can sample injection
    # parameters from the distributions specified in the config file
    waveform_parameter_generator = \
        WaveformParameterGenerator(config_file=[waveform_params_file_path],
                                   random_seed=random_seed)

    # Wrap it in a generator expression so that we can we can  simply sample
    # from it by calling next(waveform_parameters)
    waveform_parameters = \
        (waveform_parameter_generator.draw() for _ in iter(int, 1))

    # -------------------------------------------------------------------------
    # Read in static_args and variable_args defined in the config file
    # -------------------------------------------------------------------------

    # Set up a parser for the PyCBC config file
    workflow_config_parser = \
        WorkflowConfigParser(configFiles=[waveform_params_file_path])

    # Read in the PyCBC config file and amend the static_args
    variable_arguments, static_arguments = \
        read_params_from_config(workflow_config_parser)
    static_arguments = amend_static_arguments(static_arguments)

    # Ensure that static_arguments have the correct types (i.e., not str)
    static_arguments = typecast_static_args(static_arguments)

    # -------------------------------------------------------------------------
    # Define a function to generate arguments for the simulation
    # -------------------------------------------------------------------------

    def generate_arguments(injection=True):

        # Only sample waveform parameters if we are making an injection
        waveform_params = next(waveform_parameters) if injection else None

        # Return all necessary arguments as a dictionary
        return dict(static_arguments=static_arguments,
                    event_tuple=next(noise_times),
                    delta_t=config['delta_t'],
                    waveform_params=waveform_params)

    # -------------------------------------------------------------------------
    # Finally: Create our samples!
    # -------------------------------------------------------------------------

    # Keep track of all the samples we have generated
    injection_samples = []
    noise_samples = []
    injection_parameters = []

    # The procedure for generating samples with and without injections is
    # mostly the same; the only real difference is which arguments_generator
    # we have have to use:
    for sample_type in ('injections', 'noise'):
        
        # Define some sample_type-specific shortcuts
        if sample_type == 'injections':
            print('Generating samples containing an injection...')
            n_samples = config['n_injection_samples']
            arguments_generator = \
                (generate_arguments(injection=True) for _ in iter(int, 1))
        else:
            print('Generating samples *not* containing an injection...')
            n_samples = config['n_noise_samples']
            arguments_generator = \
                (generate_arguments(injection=False) for _ in iter(int, 1))

        # If we do not need to generate any samples, skip ahead:
        if n_samples == 0:
            print('Done! (n_samples=0)\n')
            continue

        # Initialize a Queue and fill it with as many arguments as we
        # want to generate samples
        arguments_queue = JoinableQueue()
        for i in range(n_samples):
            arguments_queue.put(next(arguments_generator))

        # Initialize a Queue and a list to store the generated samples
        results_queue = JoinableQueue()
        results_list = []

        # Use a tqdm context manager for the progress bar
        tqdm_args = dict(total=n_samples, ncols=80, unit='sample')
        with tqdm(**tqdm_args) as progressbar:

            # Keep track of all running processes
            list_of_processes = []

            # While we haven't produced as many results as desired, keep going
            while len(results_list) < n_samples:
    
                # -------------------------------------------------------------
                # Loop over processes to see if anything got stuck
                # -------------------------------------------------------------
                
                for process_dict in list_of_processes:
        
                    # Get start time and process object
                    start_time = process_dict['start_time']
                    process = process_dict['process']
        
                    # If the process is still running, but should have
                    # terminated already (use default limit of 60 seconds):
                    if process.is_alive() and (time.time() - start_time > 60):
            
                        # Kill process that's been running too long
                        process.terminate()
                        process.join()
                        list_of_processes.remove(process_dict)
            
                        # Add new arguments to queue
                        new_arguments = next(arguments_generator)
                        arguments_queue.put(new_arguments)
        
                    # If process has terminated already
                    elif not process.is_alive():
            
                        # If the process failed, add new arguments to queue
                        if process.exitcode != 0:
                            new_arguments = next(arguments_generator)
                            arguments_queue.put(new_arguments)
            
                        # Remove process from the list of running processes
                        list_of_processes.remove(process_dict)
    
                # Start new processes until the arguments_queue is empty, or
                # we have reached the maximum number of processes
                while (arguments_queue.qsize() > 0 and
                       len(list_of_processes) < config['n_processes']):
                    
                    # Get arguments from queue and start new process
                    arguments = arguments_queue.get()
                    p = Process(target=queue_worker,
                                kwargs=dict(arguments=arguments,
                                            results_queue=results_queue,
                                            generate_sample=generate_sample))
        
                    # Remember this process and its starting time and start it
                    list_of_processes.append(dict(process=p,
                                                  start_time=time.time()))
                    p.start()
        
                # Move stuff from the results_queue so that workers finish
                # (Otherwise the results_queue blocks the worker processes.)
                while results_queue.qsize() > 0:
                    results_list.append(results_queue.get())

                # Update the progress bar based on the number of results
                progressbar.update(len(results_list) - progressbar.n)

                # Sleep for one second before we check the processes again
                time.sleep(1)
            
        # ---------------------------------------------------------------------
        # Process results in the results_list
        # ---------------------------------------------------------------------

        # Cast results queue to list and unzip samples and parameters.
        # Note: Applying queue_to_list will empty results_queue! (And this
        # is necessary for the sample generation script to finish!)
        samples, injection_parameters_ = zip(*results_list)

        # Sort the results by event time
        idx = np.argsort([_['event_time'] for _ in samples])
        samples = [samples[i] for i in idx]
        injection_parameters_ = [injection_parameters_[i] for i in idx]

        # Store results: For noise samples, we don't need to keep the list
        # of injection parameters (which are all None)
        if sample_type == 'injections':
            injection_samples = samples
            injection_parameters = injection_parameters_
        else:
            noise_samples = samples

        print('Sample generation completed!\n')

    # -------------------------------------------------------------------------
    # Compute the normalization parameters for this file
    # -------------------------------------------------------------------------

    print('Computing normalization parameters for sample...', end=' ')

    # Group samples (with and without injection) by detector
    h1_samples = [_['h1_strain'] for _ in injection_samples + noise_samples]
    l1_samples = [_['l1_strain'] for _ in injection_samples + noise_samples]

    # Convert into a single long recording that is a numpy array
    h1_samples = np.concatenate(h1_samples)
    l1_samples = np.concatenate(l1_samples)
    
    # Compute the mean and standard deviation for both detectors
    normalization_parameters = dict(h1_mean=np.mean(h1_samples),
                                    l1_mean=np.mean(l1_samples),
                                    h1_std=np.std(h1_samples),
                                    l1_std=np.std(l1_samples))
    
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Create a SampleFile dict from list of samples and save it as an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results to HDF file ...', end=' ')

    # Initialize the dictionary that we use to create a SampleFile object
    sample_file_dict = dict(command_line_arguments=command_line_arguments,
                            static_arguments=static_arguments,
                            normalization_parameters=normalization_parameters)

    # Add injection samples: For this, we have to turn a list of dicts into
    # a numpy array for every key of the dict
    injection_samples_dict = dict()
    for key in ('event_time', 'h1_strain', 'l1_strain'):
        if injection_samples:
            value = np.array([_[key] for _ in injection_samples])
        else:
            value = None
        injection_samples_dict[key] = value
    sample_file_dict['injection_samples'] = injection_samples_dict

    # Add noise samples
    noise_samples_dict = dict()
    for key in ('event_time', 'h1_strain', 'l1_strain'):
        if noise_samples:
            value = np.array([_[key] for _ in noise_samples])
        else:
            value = None
        noise_samples_dict[key] = value
    sample_file_dict['noise_samples'] = noise_samples_dict

    # Add injection parameters
    injection_parameters_dict = dict()
    other_names = ['h1_signal', 'h1_snr', 'l1_signal', 'l1_snr',
                   'scale_factor', 'nomf_snr']
    for key in list(variable_arguments) + other_names:
        if injection_parameters:
            injection_parameters_dict[key] = \
                np.array([_[key] for _ in injection_parameters])
        else:
            injection_parameters_dict[key] = None
    sample_file_dict['injection_parameters'] = injection_parameters_dict

    # Construct the path for the HDF file
    sample_file_path = os.path.join('..', 'output', config['output_file_name'])

    # Create the SampleFile object and save it to the specified HDF file
    sample_file = SampleFile(data=sample_file_dict)
    sample_file.to_hdf(file_path=sample_file_path)

    print('Done!')

    # Get file size in MB and print the result
    sample_file_size = os.path.getsize(sample_file_path) / 1024**2
    print('Size of resulting HDF file: {:.2f}MB'.format(sample_file_size))
    print('')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    # PyCBC always create a copy of the waveform parameters file, which we
    # can delete at the end of the sample generation process
    duplicate_path = os.path.join('.', config['waveform_params_file_name'])
    if os.path.exists(duplicate_path):
        os.remove(duplicate_path)

    # Print the total run time
    print('Total runtime: {:.1f} seconds!'.format(time.time() - script_start))
    print('')
