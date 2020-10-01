# _DeepSentinel_
_DeepSentinel_: a sentinel-1 and -2 sensor fusion model for semantic embedding. A Copernicus Master's Finalist supported by Descartes Labs Impact Science Programme and Microsoft AI for Earth.

## Introduction
Earth observation offers new insight into anthropogenic changes to nature, and how these changes are effecting (and are effected by) the built environment and the real economy. With the global availability of medium-resolution (\~10m) synthetic aperature radar (SAR) Sentinel-1 and multispectral Sentinel-2 imagery, machine learning can be employed to offer these insights at scale, unbiased to company- and country-level reporting.

Machine learning applications using earth observation imagery present some unique problems. Multispectral images are often confounded by interference from clouds and atmosphere conditions. And while there is ample imagery data, geospatially-localised labels are sparse, with data quality and completeness heavily geographically skewed.

_DeepSentinel_ seeks to address these problems. DeepSentinel fuses Sentinel-2 imagery with Sentinel-1 SAR imagery which is unobstructed by clouds and atmospher conditions. We are building the largest publicly available corpus of matched Sentinel-1 and -2 imagery for the purposes of self-supervised pre-training. For select geographies, we sample the best-available land-use and land-cover datasets.

Our goal is to produce pre-trained general purpose convolutional neural networks for a number of use cases, see Figure 1. We want DeepSentinel to be as widely accessible as possible - we are developing our training corpus and models with both DescartesLabs and Google Earth Engine samples and are developing everything open-source. Please check back regularly for updates or watch or star!


*Figure 1:* Summary of _DeepSentinel_

![alt text](deepsentinel-summary.png)


### Distribution

We are developing DeepSentinel open-source and make our training corpus and models publicly available via both Google Cloud Storage and Azure Storage requester-pays instances. The following products are currently available:

#### Training Corpuses
- *v_1k_256px_nolabels*: 1,000 Sentinel-1 + Sentinel-2 256-pixel patch size samples, sampled from the global earth land mass (except antarctica) between 2019-08-01 and 2019-09-17.
- Forthcoming - check back soon!

#### Pre-trained Models
- Forthcoming - check back soon!


## Installation

To use or contribute to this code base, please follow these instructions.


## Use

We provide a command line interface (CLI) at `cli.py`. The CLI can be used with the following parameters, see `python cli.py --help` for details.


## Acknowledgements

Acquisition Start Date and Time

MMM_OPER_MSI_L1C_DS_ssss_yyyymmddthhmmss_SYYYMMDDTHHMMSS_Nxx.yy

MMM = Mission ID: S2A or S2B
OPER = File Class (Routine Operations)
MSI = Sensor (Multi-Spectral Instrument)
L1C = Processing Level
DS = Datastrip
ssss = Site Center
yyyymmddthhmmss = Creation Date and Time
S = Validity Start Time
YYYYMMDDTHHMMSS = Acquisition Start Date and Time
Nxx.yy = Vendor Software Version/Processing Baseline