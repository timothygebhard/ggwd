"""
The "main script" of this repository: Read in a configuration file and
generate synthetic GW data according to the provided specifications.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from __future__ import print_function

import argparse
import numpy as np
import os
import sys
import time

from itertools import count
from multiprocessing import Process, Queue
from tqdm import tqdm

from utils.configfiles import read_ini_config, read_json_config
from utils.hdffiles import NoiseTimeline
from utils.samplefiles import SampleFile
from utils.samplegeneration import generate_sample
from utils.waveforms import WaveformParameterGenerator


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def queue_worker(arguments, results_queue):
    """
    Helper function to generate a single sample in a dedicated process.

    Args:
        arguments (dict): Dictionary containing the arguments that are
            passed to generate_sample().
        results_queue (Queue): The queue to which the results of this
            worker / process are passed.
    """
    
    # Try to generate a sample using the given arguments and store the result
    # in the given result_queue (which is shared across all worker processes).
    try:
        result = generate_sample(**arguments)
        results_queue.put(result)
        sys.exit(0)
    
    # For some arguments, LALSuite crashes during the sample generation.
    # In this case, terminate with a non-zero exit code to make sure a new
    # set of argument is added to the main arguments_queue
    except RuntimeError:
        sys.exit('Runtime Error')


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Disable file locking for HDF files
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    # Start the stopwatch
    script_start = time.time()

    print('')
    print('GENERATE A GW DATA SAMPLE FILE')
    print('')
    
    # -------------------------------------------------------------------------
    # Parse the command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser and add arguments
    parser = argparse.ArgumentParser(description='Generate a GW data sample.')
    parser.add_argument('--config-file',
                        help='Name of the JSON configuration file which '
                             'controls the sample generation process.',
                        default='default.json')

    # Parse the arguments that were passed when calling this script
    print('Parsing command line arguments...', end=' ')
    command_line_arguments = vars(parser.parse_args())
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in JSON config file specifying the sample generation process
    # -------------------------------------------------------------------------

    # Build the full path to the config file
    json_config_name = command_line_arguments['config_file']
    json_config_path = os.path.join('.', 'config_files', json_config_name)
    
    # Read the JSON configuration into a dict
    print('Reading and validating in JSON configuration file...', end=' ')
    config = read_json_config(json_config_path)
    print('Done!')

    # -------------------------------------------------------------------------
    # Read in INI config file specifying the static_args and variable_args
    # -------------------------------------------------------------------------

    # Build the full path to the waveform params file
    ini_config_name = config['waveform_params_file_name']
    ini_config_path = os.path.join('.', 'config_files', ini_config_name)

    # Read in the variable_arguments and static_arguments
    print('Reading and validating in INI configuration file...', end=' ')
    variable_arguments, static_arguments = read_ini_config(ini_config_path)
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Shortcuts and random seed
    # -------------------------------------------------------------------------

    # Set the random seed for this script
    np.random.seed(config['random_seed'])

    # Define some useful shortcuts
    random_seed = config['random_seed']
    max_runtime = config['max_runtime']
    bkg_data_dir = config['background_data_directory']

    # -------------------------------------------------------------------------
    # Construct a generator for sampling waveform parameters
    # -------------------------------------------------------------------------

    # Initialize a waveform parameter generator that can sample injection
    # parameters from the distributions specified in the config file
    waveform_parameter_generator = \
        WaveformParameterGenerator(config_file=ini_config_path,
                                   random_seed=random_seed)

    # Wrap it in a generator expression so that we can we can easily sample
    # from it by calling next(waveform_parameters)
    waveform_parameters = \
        (waveform_parameter_generator.draw() for _ in iter(int, 1))

    # -------------------------------------------------------------------------
    # Construct a generator for sampling valid noise times
    # -------------------------------------------------------------------------

    # If the 'background_data_directory' is None, we will use synthetic noise
    if config['background_data_directory'] is None:

        print('Using synthetic noise! (background_data_directory = None)\n')

        # Create a iterator that returns a fake "event time", which we will
        # use as a seed for the RNG to ensure the reproducibility of the
        # generated synthetic noise.
        # For the HDF file path that contains that time, we always yield
        # None, so that we know that we need to generate synthetic noise.
        noise_times = ((1000000000 + _, None) for _ in count())

    # Otherwise, we set up a timeline object for the background noise, that
    # is, we read in all HDF files in the raw_data_directory and figure out
    # which parts of it are useable (i.e., have the right data quality and
    # injection bits set as specified in the config file).
    else:

        print('Using real noise from LIGO recordings! '
              '(background_data_directory = {})'.format(bkg_data_dir))
        print('Reading in raw data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = NoiseTimeline(background_data_directory=bkg_data_dir,
                                       random_seed=random_seed)

        # Create a noise time generator so that can sample valid noise times
        # simply by calling next(noise_time_generator)
        delta_t = int(static_arguments['noise_interval_width'] / 2)
        noise_times = (noise_timeline.sample(delta_t=delta_t,
                                             dq_bits=config['dq_bits'],
                                             inj_bits=config['inj_bits'],
                                             return_paths=True)
                       for _ in iter(int, 1))
        
        print('Done!\n')

    # -------------------------------------------------------------------------
    # Define a convenience function to generate arguments for the simulation
    # -------------------------------------------------------------------------

    def generate_arguments(injection=True):

        # Only sample waveform parameters if we are making an injection
        waveform_params = next(waveform_parameters) if injection else None

        # Return all necessary arguments as a dictionary
        return dict(static_arguments=static_arguments,
                    event_tuple=next(noise_times),
                    waveform_params=waveform_params)

    # -------------------------------------------------------------------------
    # Finally: Create our samples!
    # -------------------------------------------------------------------------

    # Keep track of all the samples (and parameters) we have generated
    samples = dict(injection_samples=[], noise_samples=[])
    injection_parameters = dict(injection_samples=[], noise_samples=[])

    # The procedure for generating samples with and without injections is
    # mostly the same; the only real difference is which arguments_generator
    # we have have to use:
    for sample_type in ('injection_samples', 'noise_samples'):
    
        # ---------------------------------------------------------------------
        # Define some sample_type-specific shortcuts
        # ---------------------------------------------------------------------
        
        if sample_type == 'injection_samples':
            print('Generating samples containing an injection...')
            n_samples = config['n_injection_samples']
            arguments_generator = \
                (generate_arguments(injection=True) for _ in iter(int, 1))
            
        else:
            print('Generating samples *not* containing an injection...')
            n_samples = config['n_noise_samples']
            arguments_generator = \
                (generate_arguments(injection=False) for _ in iter(int, 1))

        # ---------------------------------------------------------------------
        # If we do not need to generate any samples, skip ahead:
        # ---------------------------------------------------------------------

        if n_samples == 0:
            print('Done! (n_samples=0)\n')
            continue

        # ---------------------------------------------------------------------
        # Initialize queues for the simulation arguments and the results
        # ---------------------------------------------------------------------

        # Initialize a Queue and fill it with as many arguments as we
        # want to generate samples
        arguments_queue = Queue()
        for i in range(n_samples):
            arguments_queue.put(next(arguments_generator))

        # Initialize a Queue and a list to store the generated samples
        results_queue = Queue()
        results_list = []

        # ---------------------------------------------------------------------
        # Use process-based multiprocessing to generate samples in parallel
        # ---------------------------------------------------------------------

        # Use a tqdm context manager for the progress bar
        tqdm_args = dict(total=n_samples, ncols=80, unit='sample')
        with tqdm(**tqdm_args) as progressbar:

            # Keep track of all running processes
            list_of_processes = []

            # While we haven't produced as many results as desired, keep going
            while len(results_list) < n_samples:
    
                # -------------------------------------------------------------
                # Loop over processes to see if anything finished or got stuck
                # -------------------------------------------------------------
                
                for process_dict in list_of_processes:
        
                    # Get the process object and its current runtime
                    process = process_dict['process']
                    runtime = time.time() - process_dict['start_time']
        
                    # Check if the process is still running when it should
                    # have terminated already (according to max_runtime)
                    if process.is_alive() and (runtime > max_runtime):
            
                        # Kill process that's been running too long
                        process.terminate()
                        process.join()
                        list_of_processes.remove(process_dict)
            
                        # Add new arguments to queue to replace the failed ones
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

                # -------------------------------------------------------------
                # Start new processes if necessary
                # -------------------------------------------------------------
    
                # Start new processes until the arguments_queue is empty, or
                # we have reached the maximum number of processes
                while (arguments_queue.qsize() > 0 and
                       len(list_of_processes) < config['n_processes']):
                    
                    # Get arguments from queue and start new process
                    arguments = arguments_queue.get()
                    p = Process(target=queue_worker,
                                kwargs=dict(arguments=arguments,
                                            results_queue=results_queue))
        
                    # Remember this process and its starting time
                    process_dict = dict(process=p, start_time=time.time())
                    list_of_processes.append(process_dict)
                    
                    # Finally, start the process
                    p.start()

                # -------------------------------------------------------------
                # Move results from results_queue to results_list
                # -------------------------------------------------------------

                # Without this part, the results_queue blocks the worker
                # processes so that they won't terminate
                while results_queue.qsize() > 0:
                    results_list.append(results_queue.get())

                # Update the progress bar based on the number of results
                progressbar.update(len(results_list) - progressbar.n)

                # Sleep for some time before we check the processes again
                time.sleep(0.5)
            
        # ---------------------------------------------------------------------
        # Process results in the results_list
        # ---------------------------------------------------------------------

        # Separate the samples and the injection parameters
        samples[sample_type], injection_parameters[sample_type] = \
            zip(*results_list)

        # Sort all results by the event_time
        idx = np.argsort([_['event_time'] for _ in list(samples[sample_type])])
        samples[sample_type] = \
            list([samples[sample_type][i] for i in idx])
        injection_parameters[sample_type] = \
            list([injection_parameters[sample_type][i] for i in idx])

        print('Sample generation completed!\n')

    # -------------------------------------------------------------------------
    # Compute the normalization parameters for this file
    # -------------------------------------------------------------------------

    print('Computing normalization parameters for sample...', end=' ')

    # Gather all samples (with and without injection) in one list
    all_samples = list(samples['injection_samples'] + samples['noise_samples'])

    # Group all samples by detector
    h1_samples = [_['h1_strain'] for _ in all_samples]
    l1_samples = [_['l1_strain'] for _ in all_samples]

    # Stack recordings along first axis
    h1_samples = np.vstack(h1_samples)
    l1_samples = np.vstack(l1_samples)
    
    # Compute the mean and standard deviation for both detectors as the median
    # of the means / standard deviations for each sample. This is more robust
    # towards outliers than computing "global" parameters by concatenating all
    # samples and treating them as a single, long time series.
    normalization_parameters = \
        dict(h1_mean=np.median(np.mean(h1_samples, axis=1), axis=0),
             l1_mean=np.median(np.mean(l1_samples, axis=1), axis=0),
             h1_std=np.median(np.std(h1_samples, axis=1), axis=0),
             l1_std=np.median(np.std(l1_samples, axis=1), axis=0))
    
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Create a SampleFile dict from list of samples and save it as an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results to HDF file ...', end=' ')

    # Initialize the dictionary that we use to create a SampleFile object
    sample_file_dict = dict(command_line_arguments=command_line_arguments,
                            injection_parameters=dict(),
                            injection_samples=dict(),
                            noise_samples=dict(),
                            normalization_parameters=normalization_parameters,
                            static_arguments=static_arguments)

    # Collect and add samples (with and without injection)
    for sample_type in ('injection_samples', 'noise_samples'):
        for key in ('event_time', 'h1_strain', 'l1_strain'):
            if samples[sample_type]:
                value = np.array([_[key] for _ in list(samples[sample_type])])
            else:
                value = None
            sample_file_dict[sample_type][key] = value

    # Collect and add injection_parameters (ignore noise samples here, because
    # for those, the injection_parameters are always None)
    other_keys = ['h1_signal', 'h1_snr', 'l1_signal', 'l1_snr', 'scale_factor']
    for key in list(variable_arguments + other_keys):
        if injection_parameters['injection_samples']:
            value = np.array([_[key] for _ in
                              injection_parameters['injection_samples']])
        else:
            value = None
        sample_file_dict['injection_parameters'][key] = value

    # Construct the path for the output HDF file
    output_dir = os.path.join('.', 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sample_file_path = os.path.join(output_dir, config['output_file_name'])

    # Create the SampleFile object and save it to the specified output file
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
