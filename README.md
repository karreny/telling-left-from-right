# YouTube ASMR Upmixing Model Demo

This repository contains a demo model for audio spatialization of Youtube-ASMR videos.

From K. Yang, B. Russell and J. Salamon, "Telling Left from Right: Learning Spatial Correspondence of Sight and Sound", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Virtual Conference, June 2020.

## Usage

### Download model checkpoint

Download the model checkpoint from Google Drive ([link](https://drive.google.com/file/d/13rubXfiJGW09ptdDsSoIecN-jnu6BRSg/view?usp=sharing)).
Place this file, `upmixing-final-exp-1-flip-checkpoint-best.pth.tar`, in a directory called `models` in the repository.

### Setup
Install dependencies using [Anaconda](https://docs.conda.io/en/latest/miniconda.html)
```
conda env create -f environment.yml
conda activate stereolearning
```

### Run Jupyter notebook on demo video
In the command line, run:
```
jupyter notebook
```
In the browser, open the notebook called `upmixing-demo.ipynb` and run all of the cells.

### Expected output
Upmixed audio and video will be saved to a folder called `demo` in the repository.

### References
The code for the upmixing model is based on [2.5D Visual Sound](https://github.com/facebookresearch/2.5D-Visual-Sound).


