# _DeepSentinel_
_DeepSentinel_: a sentinel-1 and -2 sensor fusion model for semantic embedding. A Copernicus Master's Finalist supported by Descartes Labs Impact Science Programme and Microsoft AI for Earth.

## Introduction
Earth observation offers new insight into anthropogenic changes to nature, and how these changes are effecting (and are effected by) the built environment and the real economy. With the global availability of medium-resolution (\~10m) synthetic aperature radar (SAR) Sentinel-1 and multispectral Sentinel-2 imagery, machine learning can be employed to offer these insights at scale, unbiased to company- and country-level reporting.

Machine learning applications using earth observation imagery present some unique problems. Multispectral images are often confounded by interference from clouds and atmosphere conditions. And while there is ample imagery data, geospatially-localised labels are sparse, with data quality and completeness heavily geographically skewed.

_DeepSentinel_ seeks to address these problems. DeepSentinel fuses Sentinel-2 imagery with Sentinel-1 SAR imagery which is unobstructed by clouds and atmospher conditions. We are building the largest publicly available corpus of matched Sentinel-1 and -2 imagery for the purposes of self-supervised pre-training. For select geographies, we sample the best-available land-use and land-cover datasets.

Our goal is to produce pre-trained general purpose convolutional neural networks for a number of use cases, see Figure 1. We want DeepSentinel to be as widely accessible as possible - we are developing our training corpus and models with both DescartesLabs and Google Earth Engine samples and are developing everything open-source. Please check back regularly for updates or watch or star!


*Figure 1:* Summary of _DeepSentinel_

![alt text](deepsentinel-summary.png)

More details can be found at our ESA Phi-Week presentation [here](https://docs.google.com/presentation/d/1uWnbfVeZz21IY59E2RCHbfM-f7V5-xafEsuKpdTVAAE/edit?usp=sharing).


### Distribution

We are developing DeepSentinel open-source and make our training corpus and models publicly available via both Google Cloud Storage and Azure Storage requester-pays instances. The following products are currently available:

#### Training Corpuses
- *v_1k_256px_nolabels*: 1,000 Sentinel-1 + Sentinel-2 256-pixel patch size samples, sampled from the global earth land mass (except antarctica) between 2019-08-01 and 2019-09-17.
- Forthcoming - check back soon!

#### Pre-trained Models
- Forthcoming - check back soon!


## Installation


### Environment, Repo, and Packages

To use or contribute to this code base, please follow these instructions.

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management. Download and install Miniconda:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    sh ./Miniconda3-latest-Linux-x86_64.sh

Create a new environment (with python 3.6 to resolve for compatibility issues between packages) and activate it:

    conda create -n deepsentinel python=3.6

    conda activate deepsentinel

Clone this repo and enter it:

    git clone https://github.com/Lkruitwagen/deepsentinel.git

    cd deepsentinel

Install the requirements. To get the substantial performance improvements of combining pygeos and geopandas, we'll install them with Conda.

    conda install -c conda-forge --file conda_reqs.txt

    pip install -r pip_reqs.txt

### Credentials

This repo makes extensive use of third party services. 

#### Copernicus Open Access Hub

Copernicus Open Access Hub is used to access Sentinel catalog data. Obtain credentials from https://scihub.copernicus.eu/dhus/#/self-registration and save them in a `json` with `json.dump('{"scihub":{"U":"<yourusername>","P":"<yourpassword>"}}', open('<path/to/credentials.json>','w'))`. Edit the path in `CONFIG.yaml` and `bin/make_config.py`.

#### Descartes Labs

Descartes Labs is used to obtain Sentinel-1 and Sentinel-2 data. Contact [DescartesLabs](https://www.descarteslabs.com/) for platform access and then use `descarteslabs auth login` to log in and save the access token to your `$HOME` directory. 

#### Google Earth Engine

Google Earth Engine is used to obtain Sentinel-1 and Sentinel-2 data. Sign up for Earth Engine [here](https://earthengine.google.com/new_signup/). You will need REST API access, for which you may need to contact Earth Engine support. Create a service account for use with the REST API following instructions [here](https://developers.google.com/earth-engine/guides/service_account). Edit the path to your Earth Engine `json` credentials in `CONFIG.yaml` and `bin/make_config.py`.

#### Google Cloud Storage

To use your own google cloud storage bucket with _DeepSentinel_, create your own storage bucket, and then create a service account and `json` key, following instructions [here](https://cloud.google.com/iam/docs/creating-managing-service-account-keys). Edit the path to your GCP `json` credentials in `CONFIG.yaml` and `bin/make_config.py`.

#### Azure Cloud Storage

To use your own Azure cloud storage account with _DeepSentinel_, create your own storage account, and then obtain a connection string for it, copying `Connection string` of `key1` under the Access keys tab for your storage account. Save the string in a txt file and edit the path to your connection string file in `CONFIG.yaml` and `bin/make_config.py`.

## Acknowledgements

We are extremely grateful for the ongoing support of [DescartesLabs Impact Science Programme](https://www.descarteslabs.com/impact_science/) and [Microsoft AI for Earth](https://www.microsoft.com/en-us/ai/ai-for-earth) programme.

### tensorboard

To use tensorboard to view experiments, run:

    tensorboard --logdir /path/to/deepsentinel/experiments/tensorboard
    
This should start tensorboard running, usually at `localhost:6006`. To view tensorboard in your browser, create a remote ssh tunnel to your machine:

    ssh -N -f -L localhost:6006:localhost:6006 <username>@<ip-address>
    
Or on gcloud (you may need to be logged in with `gcloud auth login`):

    gcloud beta compute ssh --zone "<your-instance-zone>" "<your-instance-name>" --project "<your-instance-project>" -- -L 6006:localhost:6006

    