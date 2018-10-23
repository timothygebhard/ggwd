"""
Provide command line parsers for scripts.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class DefaultParser:

    def __init__(self):
        """
        Initialize the parser and define possible options.
        """

        # Set up a new parser for command line arguments
        self.parser = argparse.ArgumentParser()

        # Add all the accepted / necessary options
        self.parser.add_argument('--config-file',
                                 help='Name of the JSON configuration file.',
                                 default='default.json')

    # -------------------------------------------------------------------------

    def parse(self):
        """
        Parse and pre-process the provided command line arguments.

        Returns: dict
            A dictionary containing the options that were provided as
            command line arguments.
        """

        # Actually parse the arguments provided when calling the script
        arguments = vars(self.parser.parse_args())

        return arguments
