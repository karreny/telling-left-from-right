# YouTube ASMR Download (forked from [marl/audiosetdl](https://github.com/marl/audiosetdl))

This repository contains scripts for downloading ASMR video clips from the YouTube-ASMR dataset.

The YouTube-ASMR-300K dataset contains URLS for over 900 hours of ASMR video clips with stereo/binaural audio produced by various YouTube artists. A clean subset of the dataset (YouTube-ASMR) contains ASMR video clips that were partially manually curated. The following paper contains a detailed description of the dataset and how it was compiled:

K. Yang, B. Russell and J. Salamon, "Telling Left from Right: Learning Spatial Correspondence of Sight and Sound", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Virtual Conference, June 2020.

## Usage

### Download URLs

Download the URLS from Zenodo ([link](https://zenodo.org/record/3889168)) and unzip the file to the repo directory.
i.e.
```
wget -O youtube_asmr.zip https://zenodo.org/record/3889168/files/youtube_asmr.zip?download=1
unzip youtube_asmr.zip
```

### Setup
Install dependencies using [Anaconda](https://docs.conda.io/en/latest/miniconda.html)
```
conda env create -f environment.yml
conda activate yt-asmr
```

### Run download script
Run the download script, specifying the URL file (url_file_path) and the output directory (data_dir)
```
python download_yt_asmr.py <url_file_path> <data_dir>
```
For example, to download the YouTube-ASMR test set to a directory called `test_data`, run:
```
python download_yt_asmr.py youtube_asmr/test.csv test_data
```
### Expected output
MP4 video clips and FLAC audio clips will be downloaded to `<data_dir>/data/video/` and `<data_dir>/data/audio` respectively. 
Download details (including errors) will be logged at `yt-asmr-dl.log`.


