"""
Provide tools for multiprocessing with Queues.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from multiprocessing import Lock


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ThreadsafeIter:
    """
    Provide a wrapper to make iterables thread-safe in a multiprocessing
    environment.
    """

    def __init__(self, iterable):

        self.iterable = iterable
        self.lock = Lock()

    def __iter__(self):

        return self

    def next(self):

        with self.lock:
            return self.iterable.next()


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def queue_worker(arguments_queue,
                 results_queue,
                 arguments_generator,
                 generate_sample,
                 progressbar):
    """
    This function will be passed to a queue worker and is responsible for
    getting a set of arguments from the arguments queue, generating the
    corresponding sample, adding it to the results queue, and updating the
    progress bar.

    Args:
        arguments_queue: JoinableQueue
            A queue containing dictionaries of the form {static_args,
            event_time, waveform_params}. These are the arguments that are
            passed to the generate_sample() function.
        results_queue: JoinableQueue
            A queue containing the results of the samples that this queue
            worker has produced
        arguments_generator: generator
            A generator that can be used to generate new arguments in case
            the sample generation fails for a certain set of arguments
        generate_sample: function
            A function that can generate samples. Usually, this should be
            generate_sample(static_arguments, event_time, waveform_params)
            as defined in WaveformTools.py
        progressbar: iterator
            A tqdm-decorated iterator that realizes the progress bar
    """

    while True:

        # Retrieve the arguments for the next sample
        arguments = arguments_queue.get()

        # If they are None, we can close this worker
        if arguments is None:
            arguments_queue.task_done()
            break

        # Otherwise we try to generate a sample
        try:
            result = generate_sample(**arguments)

        # For some arguments, LALSuite crashes during the sample generation.
        # In this case we can try again with different waveform parameters:
        except RuntimeError:

            # Remove the current (failed) task from the queue
            arguments_queue.task_done()

            # Generate a new set of arguments / waveform parameters
            arguments = next(arguments_generator)

            # Put the updated arguments back in the queue
            arguments_queue.put(arguments)
            continue

        # If the sample generation succeeded, update the progress bar,
        # save the result, and remove the finished task from the queue
        results_queue.put(result)
        progressbar.update(results_queue.qsize() - progressbar.n)
        arguments_queue.task_done()


# -----------------------------------------------------------------------------


def queue_to_list(queue):
    """
    Take a multiprocessing.Queue and cast it to list. Note that applying
    this function will empty the queue that is passed to queue_to_list()!

    Args:
        queue (multiprocessing.Queue): A Queue object.

    Returns:
        (list): The queue casted to list.
    """

    # Store the result
    results_list = []

    # Get elements from queue until the queue is empty and append them to
    # the list that we are going to return
    while queue.qsize() > 0:
        results_list.append(queue.get())

    return results_list
