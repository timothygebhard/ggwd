How to get real LIGO data
=========================

.. note::
   This step is optional: If you only want to work with simulated LIGO 
   noise, you can skip this part.

As a next step, you will need to download the LIGO recordings, which will be 
used for the background noise from the `Gravitational Wave Open Science Center
(GWOSC) <https://www.gw-openscience.org/archive/O1>`_, formerly known as LOSC. 
We recommend you to download the data at a sampling frequency of 4096 Hz. 
As the file format, choose HDF5. 
GWOSC also has `a tutorial on how to download data 
<https://www.gw-openscience.org/tutorial01/>`_, which is a bit out of date, 
but may be still of use.

However, to help you get started with the batch download, the root directory 
also contains a script named ``download_gwosc_data.py``, which you can use as 
follows to start downloading data:

.. code-block:: bash

   python download_gwosc_data.py --destination=/path/to/download/folder

At the time when the code for this repository was first written, only the 
data for Observation Run 1 (O1) had been released to the public, and some 
of the default values of the download script (e.g., the start and end time 
stamps) are also based on O1. 

.. note::
   This has now changed, and the data for O2 is also available.
   Currently, the download script have not been adjusted to this yet; 
   however, this should be a pretty straight-forward fix.

The download script will automatically create folders for the two detectors 
``H1`` and ``L1`` in the specified target directory, and will sort the files 
accordingly. 
Due to different detector downtimes, the number of HDF files ending up in 
these two directories is *not* expected to be the same!

Note that ``download_gwosc_data.py`` also has a ``--dry`` option, in case you 
want to check what the script *would* download without actually downloading 
anything yet.
Furthermore, if you are actually interested in what is inside such an HDF 
file, you may want to take a look at the `What's in a LIGO data file?
<https://www.gw-openscience.org/tutorial02>`_ tutorial on GWOSC, which 
explains the internal structure of these files and how to extract data 
yourself.

.. warning::
   The full LIGO recordings for the first observation run (O1) consist of 
   approximately 361 GB worth of HDF files, so be sure you have enough 
   storage! 
   Also, depending on your internet connection, the download will probably 
   take several hours :-)

