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
import time
import sys
import os
import numpy as np

from tqdm import tqdm
from multiprocessing import Process, JoinableQueue

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config

# Here we need to add the parent directory to the $PYTHONPATH, because
# apparently Python does not have a less hacky way of importing from a
# sibling directory if you are "just" in a script and not in a package
sys.path.insert(0, os.path.realpath('..'))

# Now we can even import from generate_gw_data without PyCharm complaining!
from generate_gw_data.HDFTools import NoiseTimeline  # noqa
from generate_gw_data.MultiprocessingTools import ThreadsafeIter, \
    queue_worker, queue_to_list  # noqa
from generate_gw_data.WaveformTools import WaveformParameterGenerator, \
    generate_sample, amend_static_arguments  # noqa
from generate_gw_data.SampleFileTools import SampleFile  # noqa
from generate_gw_data.TypecastingTools import typecast_static_args  # noqa


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
    script_start_time = time.time()
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
    assert os.path.exists(config_file_path), \
        'Specified configuration file does not exist!'

    # Read the configuration into a dict
    print('Reading and validating in configuration file...', end=' ')
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # Make sure all required keys are present
    for key in ('background_data_directory', 'dq_bits', 'inj_bits',
                'waveform_params_file_name', 'n_injection_samples',
                'n_noise_samples', 'n_processes', 'random_seed',
                'output_file_name'):
        assert key in config.keys(), \
            'Missing key: {}'.format(key)
    print('Done!')
    print()

    # -------------------------------------------------------------------------
    # Ensure the waveform parameters file exists
    # -------------------------------------------------------------------------

    # Construct the path to the waveform params file
    waveform_params_file_name = config['waveform_params_file_name']
    waveform_params_file_path = \
        os.path.join('..', 'config_files', waveform_params_file_name)

    # Ensure it exists
    assert os.path.exists(waveform_params_file_path), \
        'Specified waveform parameter file does not exist!'

    # -------------------------------------------------------------------------
    # Construct a generator for sampling valid noise times
    # -------------------------------------------------------------------------

    # Define some shortcuts
    random_seed = config['random_seed']
    background_data_directory = config['background_data_directory']

    # Set the random seed for this script
    np.random.seed(config['random_seed'])

    # If the 'background_data_directory' is None, we will use synthetic noise.
    # In this case, there are no noise times (always return None).
    if background_data_directory is None:

        print('background_data_directory is None -> Using synthetic noise!')

        # Create a ThreadsafeIter that returns a fixed fake "event time".
        # This is necessary, because otherwise we run into all sorts of issues
        # with PyCBC TimeSeries objects. However, for the HDF file path that
        # contains that time, we return None, so that we now that we need to
        # generate synthetic noise.
        noise_times = ThreadsafeIter((1234567890, None) for _ in iter(int, 1))

    # Otherwise, we set up a timeline object for the background noise, that
    # is, we read in all HDF files in the raw_data_directory and figure out
    # which parts of it are useable (i.e., have the right data quality and
    # injection bits set as specified in the config file).
    else:

        print('Reading in raw data. This may take several minutes...', end=' ')

        # Create a timeline object by running over all HDF files once
        noise_timeline = \
            NoiseTimeline(background_data_directory=background_data_directory,
                          random_seed=random_seed)

        # Create a noise time generator so that can sample valid noise times
        # simply by calling next(noise_time_generator)
        noise_times = \
            ThreadsafeIter((noise_timeline.sample(delta_t=config['delta_t'],
                                                  dq_bits=config['dq_bits'],
                                                  inj_bits=config['inj_bits'],
                                                  return_paths=True)
                            for _ in iter(int, 1)))
        print('Done!')

    # -------------------------------------------------------------------------
    # Construct a generator for sampling waveform parameters
    # -------------------------------------------------------------------------

    # Initialize a waveform parameter generator that can sample injection
    # parameters from the distributions specified in the config file
    waveform_parameter_generator = \
        WaveformParameterGenerator(config_file=[waveform_params_file_path],
                                   random_seed=random_seed)

    # Wrap it in a thread-safe generator expression so that we can we can
    # simply sample from it by calling next(waveform_parameters)
    waveform_parameters = ThreadsafeIter((waveform_parameter_generator.draw()
                                          for _ in iter(int, 1)))

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
    # Set up an argument generator for samples with/without injections
    # -------------------------------------------------------------------------

    # Set up a function that generates the arguments for generate_sample():
    # We need the static_arguments, a (random) event time, and some (random)
    # waveform parameters for the simulation
    def generate_arguments(make_injection=True):

        # Depending on whether we want to make an injection or not,
        # we either sample a set of parameters or not
        if make_injection:
            waveform_params = next(waveform_parameters)
        else:
            waveform_params = None

        # Return the dict of arguments
        return dict(static_arguments=static_arguments,
                    event_tuple=next(noise_times),
                    delta_t=config['delta_t'],
                    waveform_params=waveform_params)

    # Create thread-safe generator expressions for the arguments_generator()
    arguments_generator_injections = \
        ThreadsafeIter((generate_arguments(make_injection=True)
                        for _ in iter(int, 1)))
    arguments_generator_noise = \
        ThreadsafeIter((generate_arguments(make_injection=False)
                        for _ in iter(int, 1)))

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
    for loop_dict in [dict(make_injection=True,
                           n_samples=config['n_injection_samples'],
                           arguments_generator=arguments_generator_injections,
                           print_string=''),
                      dict(make_injection=False,
                           n_samples=config['n_noise_samples'],
                           arguments_generator=arguments_generator_noise,
                           print_string='*not* ')]:

        # Unpack the loop dict
        make_injection = loop_dict['make_injection']
        n_samples = loop_dict['n_samples']
        arguments_generator = loop_dict['arguments_generator']
        print_string = loop_dict['print_string']

        # Print what kind of samples we are now generating
        print('Generating samples {}containing an injection...'.
              format(print_string))

        # If we do not need to generate any samples, skip ahead:
        if n_samples == 0:
            print('Done! (n_samples=0)\n')
            continue

        # Initialize a Queue and fill it with as many arguments as we
        # want to generate samples
        arguments_queue = JoinableQueue()
        for item in range(n_samples):
            arguments_queue.put(next(arguments_generator))

        # Initialize a Queue to store the results of the sample generation
        results_queue = JoinableQueue()

        # Set up parameters for progressbar
        tqdm_args = dict(total=n_samples, ncols=80, unit='sample')

        with tqdm(**tqdm_args) as progressbar:

            # Start n_processes new processes to process the arguments_queue
            for i in range(config['n_processes']):

                # Collect the keyword arguments that need to be passed to the
                # queue_worker() function
                kwargs = dict(arguments_queue=arguments_queue,
                              results_queue=results_queue,
                              arguments_generator=arguments_generator,
                              generate_sample=generate_sample,
                              progressbar=progressbar)

                # Start a new process that runs a queue_worker with the
                # arguments we just collected
                process = Process(target=queue_worker, kwargs=kwargs)
                process.daemon = True
                process.start()

            # Wait until the queue is empty
            arguments_queue.join()

            # End the queue_worker processes for the arguments_queue
            for i in range(config['n_processes']):
                arguments_queue.put(None)
            arguments_queue.join()

        # Cast results queue to list and unzip samples and parameters.
        # Note: Applying queue_to_list will empty results_queue! (And this
        # is necessary for the sample generation script to finish!)
        results_list = queue_to_list(results_queue)
        samples, injection_parameters_ = zip(*results_list)

        # Sort the results by event time
        idx = np.argsort([_['event_time'] for _ in samples])
        samples = [samples[i] for i in idx]
        injection_parameters_ = [injection_parameters_[i] for i in idx]

        # Store results: For noise samples, we don't need to keep the list
        # of injection parameters (which are all None)
        if make_injection:
            injection_samples = samples
            injection_parameters = injection_parameters_
        else:
            noise_samples = samples

        print('Generation of samples {}containing injections completed!'.
              format(print_string))
        print('')

    # -------------------------------------------------------------------------
    # Create a SampleFile dict from list of samples and save it as an HDF file
    # -------------------------------------------------------------------------

    print('Saving the results ...', end=' ')

    # Initialize the dictionary that we use to create a SampleFile object
    sample_file_dict = dict(command_line_arguments=command_line_arguments,
                            static_arguments=static_arguments)

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
    print('Total runtime: {:.1f} seconds!'
          .format(time.time() - script_start_time))
    print('')
