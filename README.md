# ALTO
convLSTM for predicting large scale cloud movments.

## Table of Contents
1. [Required Libraries](#required-libraries)  
2. [Installation](#installation)
3. [Download images](#download-images)
4. [Train model](#train-model)
5. [Predict with pre-trained model in the terminal](#predict-with-pre-trained-model-in-the-terminal)
6. [Userinterface](#userinterface)

## Required Libraries
- `keras`
- `warnings`
- `owslib`
- `datetime`
- `eumdac`
- `ssl`
- `cv2`
- `numpy`
- `argparse`
- `time`
- `tqdm`
- `random`
- `copy`
- `os`
- `torch`
- `pytorch_msssim`
- `glob`
- `matplotlib`
- `fastapi`
- `pydantic`
  
## Installation
1. **Clone the repo**  
   In GIT Bash, type:  
   `git clone https://github.com/ArvidMoller/alto.git`
2. **Open in VS Code or another preferred editor**
3. **Install libraries if not already installed (showed in pip, other installers may be used)**  
   Run: `pip install [library]`
4. **Add `download_log.txt`**  
   You must add the file `download_log.txt` in `\src\satellite_imager_download\images`.

## Download images
Images for training needs to be downloaded localy from EUMETSATs API by unsing `image_download.py`
1. Navigate to `\src\satellite_imagery_download` using `cd` in the terminal.
2. Change start and end dates in the `image_download.py` file. 
3. Run: `python image_download.py`
4. Follow the instruction in the terminal.

## Train model
1. Navigate to `\src\model_training` using `cd` in the terminal.
2. Run: `python training.py`
3. Follow the instructions in the terminal.

## Predict with pre-trained model in the terminal
1. Navigate to `\src` using `cd` in the terminal.
2. Run: `python predict.py`
3. Follow the instructions in the terminal.

## Userinterface
To start server and API:
1. Navigate to `\src\api` using `cd` in the terminal.
2. Run: `fastapi dev apimain.py`
3. Start localhost or other server to show the HTML document. 
