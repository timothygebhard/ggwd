"""
Provide a custom ProgressBar class, which provides a wrapper around
an iterable that automatically produces a progressbar when iterating
over it.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import sys
import time
from threading import Event, Thread


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class RepeatedTimer:
    """
    Wrapper class to repeat the given `func` every `interval` seconds
    (asynchronously in the background).
    Source: https://stackoverflow.com/a/33054922/4100721.
    """

    def __init__(self, interval, func, *args, **kwargs):
        self.interval = interval
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target)
        self.thread.start()

    def _target(self):
        while not self.event.wait(self._time):
            self.func(*self.args, **self.kwargs)

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()


# -----------------------------------------------------------------------------


class ProgressBar:
    """
    :class:`ProgressBar` objects are a custom way to "decorate"
    a given iterable to produce a progress bar when looping over it.
    This class allows to also produce some output with the progress
    bar, such as information about the element of the iterable that is
    currently being processed.

    Args:
        iterable (iterable): The iterable to be "decorated" with
            a progressbar.
        bar_length (int): Length of the bar itself (in characters).
        auto_update (bool): Whether or not to automatically write
            the updated progressbar to the command line.
    """
    
    def __init__(self,
                 iterable,
                 bar_length=50,
                 auto_update=False):

        self.iterable = iterable
        self.max_value = len(iterable)
        self.bar_length = bar_length
        self.auto_update = auto_update

        self.start_time = None
        self.last_timediff = None

        self.progressbar = self.get_progressbar(-1)

        self.extras_ = []
        self.scheduler = None

    # -------------------------------------------------------------------------

    def __iter__(self):

        # Start the stop watch as soon as we start iterating
        self.start_time = time.time()

        # Initialize index to 0 to ensure it is always defined
        index = 0

        # Start the scheduler that will update the elapsed time every second
        def update():
            self.progressbar = self.get_progressbar(index)
            self.write(extras=self.extras_)
        self.scheduler = RepeatedTimer(1, update)

        # Actually loop over the iterable
        for index, value in enumerate(self.iterable):

            # Update the last_timediff, which is used to estimate when we
            # will be done
            self.last_timediff = self.get_timediff()

            # Update the progressbar string
            self.progressbar = self.get_progressbar(index)

            # If are doing auto-updates (i.e. no extras), we can already
            # write the progress bar to stdout
            if self.auto_update:
                self.write()

            # Finally, actually yield the current value of the iterable
            yield value

        # Update our progress bar string one last time to indicate we have
        # made it to 100%
        self.progressbar = self.get_progressbar(self.max_value)

        # Stop our background scheduler
        self.scheduler.stop()

    # -------------------------------------------------------------------------

    def get_timediff(self):
        """
        Returns: Time elapsed since progress bar was instantiated.
        """

        if self.start_time is not None:
            return time.time() - self.start_time
        else:
            return None

    # -------------------------------------------------------------------------

    def get_eta(self,
                percent):
        """
        Get the estimated time of arrival (ETA) by linear interpolation.
        
        Args:
            percent (float): Current progress in percent.

        Returns:
            Estimated time of arrival in seconds.
        """

        if self.last_timediff is not None and percent != 0:
            return max(0, self.last_timediff / percent - self.get_timediff())
        else:
            return None

    # -------------------------------------------------------------------------

    def get_progressbar(self,
                        index):
        """
        Construct the progressbar itself (bar, ETA, etc.).
        
        Args:
            index (int): Current index of the iterable; used to compute
                the current progress percentage.

        Returns:
            A string containing the basic progress bar.
        """

        # Construct the actual progress bar
        percent = float(index) / self.max_value
        bar = '=' * int(round(percent * self.bar_length))
        spaces = '-' * (self.bar_length - len(bar))

        # Get the elapsed time as a proper string
        elapsed_time = self.get_timediff()
        if elapsed_time is None:
            elapsed_time = 0
        elapsed_time = '{:.2f}'.format(elapsed_time)

        # Get the expected time of arrival (ETA) as a proper string
        eta = self.get_eta(percent)
        if eta is None:
            eta = '?'
        else:
            eta = '{:.2f}'.format(eta)

        # Construct the actual progress bar string
        out = "[{0}] {1:>3}% ({2:>{3}}/{4:>{3}}) | Elapsed: {5} | ETA: {6}"
        progressbar = out.format(bar + spaces, round(percent * 100),
                                 index, len(str(self.max_value)),
                                 self.max_value, elapsed_time, eta)

        return progressbar

    # -------------------------------------------------------------------------

    def write(self,
              clear_line=False,
              extras=()):
        """
        Construct the progress bar and write it to the command line.
        
        Args:
            clear_line (bool): Whether or not to clear the last line.
            extras (list): List of additional outputs (e.g., the file
                that is currently being downloaded).
        """

        self.extras_ = extras

        if extras:
            for _ in range(len(extras)):
                sys.stdout.write('\r\033[K\033[F')
        if clear_line:
            sys.stdout.write('\r\033[K\033[F')

        # Actually write the finished progress bar to the command line
        sys.stdout.write('\r\033[K\033[F')
        sys.stdout.write('\r\033[K\033[K')
        sys.stdout.write(self.progressbar)
        if extras:
            sys.stdout.write('\n' + '\n'.join(extras))
        if not clear_line:
            sys.stdout.write('\n')
        sys.stdout.flush()
