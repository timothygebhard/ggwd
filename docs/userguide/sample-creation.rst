.. _creating-your-first-sample:

Creating your first sample
==========================


Once you have set up the Python environment (and optionally downloaded the 
real LIGO background recordings from GWOSC), you can start doing what is the 
main purpose of this repository: 
Generate your own synthetic gravitational-wave data!

To create your first sample (with default values), simply run the following
command:

.. code-block:: bash

   python generate_sample.py --config-file=default.json


Depending how many samples your are creating (specified in the `*.json` file), 
this now may take a while (usually not more than a few minutes). 
Once the sample generation process is completed, the resulting sample file 
should eventually be stored in the ``/output`` directory.

*Ta-dah!* That's it already! Really quite simple, isn't it? :-)

Starting from here, your next steps should either be to check out how you can
:ref:`plot-results` of the sample generation process or learn more about the 
:ref:`configuration-files` that offer you fine-grained control over the entire
sample generation process.
