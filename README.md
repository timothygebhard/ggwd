# generate-gw-data

The purpose of this repository is to provide a starting point for generating—more of less—realistic samples of synthetic gravitational wave (GW) data which can be used, e.g., for machine learning experiments. By "samples", we mean time series data, which either does or does not contain a GW waveform. Since the latter have to be simulated, only samples with signals from gravitational waves originating from binary mergers can be created.

In short (for details, see below), the main idea behind the data generating process is the following: 

1. Randomly select a piece of real LIGO recording to serve as background noise, 
2. Randomly select the parameters of a compact binary coalescence (CBC) from a given parameter space and simulate the corresponding waveform,
3. Add ("inject") this simulated waveform into the background noise.

The scripts in this repository are essentially just a convenience wrapper around the [PyCBC software package](https://pycbc.org/), which itself relies partly on the [LIGO Algorithm Library (LAL)](https://wiki.ligo.org/Computing/DASWG/LALSuite). 



## 1. How to get started

In this section, we briefly describe the necessary steps to get the scripts running and to generate your first samples. More details on each step can be found in the sections below. But now, let's get started:

### 1.1 Downloading the LIGO Recordings

As a first step, you will need to download the LIGO recordings, which will be used for the background noise from the [LIGO Open Science Center (LOSC)](https://www.gw-openscience.org/archive/O1/). We recommend you to download the data at a sampling frequency of 4096 Hz. As the file format, choose HDF5. LOSC also has [a tutorial on downloading data](https://www.gw-openscience.org/tutorial01/), which is a bit out of date, but may be still of use.

To help you get started with the batch download, the `scripts` directory contains a script `download_losc_data.py`, which you can use as follows:

```shell
python download_losc_data.py --destination=/path/to/download/folder
```

At the time of writing of this guide, only the data for Observation Run 1 (O1) has been released to the public. Some of the default values of the download script (e.g., the start and end time stamps) are also based on O1. However, unless the file format is changed significantly for future releases, most of this code should be reusable for O2 with only minor edits.

The download script will automatically create folders for the two detectors  `H1` and `L1` in the specified target directory, and will sort the files accordingly. Due to different downtimes for the detectors, the number of HDF files ending up in these two directories is not expected to be the same!

Please note that `download_losc_data.py` also has a `--dry` option, in case you want to check what the script *would* download without actually downloading anything yet. Furthermore, if you are actually interested in what is inside such an HDF file, you may want to take a look at the [What's in a LIGO data file?](https://www.gw-openscience.org/tutorial02/) tutorial on LOSC, which explains the internal structure of these files and how to extract data yourself. 


Finally, a word of warning: The full LIGO recordings for O1 are about 361 GB worth of HDF files, so be sure you have enough storage! Also, depending on your internet connection, the download will probably take several hours.