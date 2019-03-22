"""
Provide tools for multiprocessing with Queues.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import sys


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
