# ggwd: generate gravitational-wave data

[![Python](https://img.shields.io/badge/Python-2.7-yellow.svg)]()
[![CodeFactor](https://www.codefactor.io/repository/github/timothygebhard/ggwd/badge)](https://www.codefactor.io/repository/github/timothygebhard/ggwd)
[![GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/timothygebhard/ggwd/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-1904.08693-red.svg)](https://arxiv.org/abs/1904.08693)
[![DOI](https://zenodo.org/badge/154147244.svg)](https://zenodo.org/badge/latestdoi/154147244)

The purpose of this repository is to provide a starting point for **generating realistic synthetic gravitational-wave data** which can then be used, for example, as training data for machine learning experiments. It was originally developed for our paper [*Convolutional neural networks: a magic bullet for gravitational-wave detection?*](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.063015) [[arXiv:1904.08693](https://arxiv.org/abs/1904.08693)] and used to generate the training and testing data for the network presented there.

More specifically, the methods provided in this repository here will help you with the simulation of GW signals from compact binary coalescences (e.g., binary black hole mergers), adding these signals into a pieces of background noise (either synthetic or real LIGO background recordings) and applying standard post-processing steps (e.g., whitening and band-passing) to results.

**Disclaimer**: The scripts here are mostly just convenience wrappers around functionalities provided by the great and mighty [PyCBC software package](https://pycbc.org/) (which itself relies in parts on the [LIGO Algorithm Library](https://wiki.ligo.org/Computing/DASWG/LALSuite)).



## Quickstart

As of now, this project is not a "proper" Python package, but only a collection of scripts, which don't need to be installed. Therefore, simply clone the repository:

```
git clone git@github.com:timothygebhard/ggwd.git ; cd ggwd
```

Ensure your Python environment fulfills all the requirements specified in `requirements.txt` (ideally, simply use a fresh virtual environment). **Please note that due to the dependence on PyCBC, this code currently only works with Python 2.7!** Now, you should be able to generate your first data sample by simply running:

```
python generate_sample.py
```

This will some default values generate a small demo sample file in the `./output` directory. Please check out the documentation to see how to adjust the configuration to your needs, and, for example, also generate samples using real LIGO recordings as the background noise.

You can then use the `plot_sample.py` script to inspect the results, which should look something like this:

![](./docs/userguide/images/sample_with_injection.png)



## Documentation

The documentation for all code, including detailed guides describing, for example, how to control the details of the sample generation process through customized configuration files, can be build simply by running `make html` in the `./docs` directory. The result will be placed in the `./docs/build/html` directory.



## Contributing to this project

Contributions to this project are very welcome! If you have found any issues with the code, or would like to contribute to add further functionality, please do not hesitate to get in touch, open an issue on GitHub, or directly send a pull request!



## Authors & License

The code in this repository is developed by [Timothy Gebhard](https://github.com/timothygebhard) and [Niki Kilbertus](https://github.com/nikikilbertus). It is distributed under the GPL-3.0 license (see [LICENSE](https://github.com/timothygebhard/ggwd/blob/master/LICENSE) file for details), which means in particular that it is provided *as is*, without any warranty, liability, or guarantees regarding its correctness.
