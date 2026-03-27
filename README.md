# ALTO
convLSTM for predicting large scale cloud movments. This project was done as a upper secondary school diploma project and a scientific poster (in Swedish), as well as sources and all results brought up in said poster (parts 7-10), are therefore linked in this readme.

## Table of Contents
1. [Required Libraries](#required-libraries)  
2. [Installation](#installation)
3. [Download images](#download-images)
4. [Train model](#train-model)
5. [Predict with pre-trained model in the terminal](#predict-with-pre-trained-model-in-the-terminal)
6. [Userinterface](#userinterface)
7. [Accuracy data](#accuracy-data)
8. [Time to generate images](#time-to-generate-images)
9. [Generated images](#generated-images)
10. [Sources for poster](#sources-for-poster)

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

## Accuracy data
The following data is collected with a model trained with the following paramiters:
- 32 filters on all layers
- Tanh activation
- 5x5, 3x3, 1x1, 3x3x3 kernel size
- MAE & SSIM coposite loss

Precision is messured as a procentage of pixels within 10 gray-values between the generated image and the corresponding satellite image. Messurements were made over 5 sets of generated images, each set containing 10 images.

A regression analysis was made on the mean precision, yeilding the following function. "x" represents the frame's number in the sequence and the function gives the expected accuracy of the frame:\
$f(x)=18,097*0,907^x$

Mean accuracy over 5 predictions:
| Number in sequence | Mean accuracy (%) |
|-------------|---------------------------|
| 1 | 18,8255 |
| 2 | 14,5765 |
| 3 | 12,83075 |
| 4 | 11,538 |
| 5 | 10,74125 |
| 6 | 9,7295 |
| 7 | 8,74125 |
| 8 | 8,2615 |
| 9 | 7,76575 |
| 10 | 7,21625 |

1st image-set:
| Number in sequence | Accuracy (%) |
|-------------|---------------|
| 1 | 14,394 |
| 2 | 11,502 |
| 3 | 10,786 |
| 4 | 10,392 |
| 5 | 10,415 |
| 6 | 10,167 |
| 7 | 9,807 |
| 8 | 9,819 |
| 9 | 9,6 |
| 10 | 8,902 |

2nd image-set:
| Number in sequence | Accuracy (%) |
|-------------|---------------|
| 1 | 24,607 |
| 2 | 18,865 |
| 3 | 16,253 |
| 4 | 14,856 |
| 5 | 13,559 |
| 6 | 12,225 |
| 7 | 10,712 |
| 8 | 9,742 |
| 9 | 8,818 |
| 10 | 7,877 |

3rd image-set:
| Number in sequence | Accuracy (%) |
|-------------|---------------|
| 1 | 24,747 |
| 2 | 18,356 |
| 3 | 15,559 |
| 4 | 13,148 |
| 5 | 11,716 |
| 6 | 10,443 |
| 7 | 8,783 |
| 8 | 7,878 |
| 9 | 7,405 |
| 10 | 7,05 |

4th image-set:
| Number in sequence | Accuracy (%) |
|-------------|---------------|
| 1 | 13,851 |
| 2 | 12,431 |
| 3 | 12,106 |
| 4 | 10,802 |
| 5 | 9,707 |
| 6 | 8,914 |
| 7 | 8,257 |
| 8 | 7,71 |
| 9 | 7,244 |
| 10 | 7,933 |

5th image-set:
| Number in sequence | Accuracy (%) |
|-------------|---------------|
| 1 | 11,554 |
| 2 | 9,583 |
| 3 | 8,725 |
| 4 | 7,756 |
| 5 | 7,275 |
| 6 | 6,083 |
| 7 | 5,663 |
| 8 | 5,607 |
| 9 | 5,24 |
| 10 | 5,036 |

## Time to generate images
Messurements were made over 5 sets of generated images, each set containing 10 images. Times were messured with the following hardware:
- CPU: AMD Ryzen 9 9900X
- GPU: Nvidia RTX 5090
- RAM: 128 GB, 5600 Hz

Mean generation-time:
| Number in sequence | Mean time (milliseconds) |
| ----------- | ------------------------------ |
| 1           | 195,4                          |
| 2           | 62,0                           |
| 3           | 70,0                           |
| 4           | 44,4                           |
| 5           | 45,4                           |
| 6           | 45,4                           |
| 7           | 44,6                           |
| 8           | 44,4                           |
| 9           | 44,8                           |
| 10          | 44,4                           |

1st image-set:
| Number in sequence | Time (milliseconds) |
| ----------- | ------------------- |
| 1           | 202                 |
| 2           | 88                  |
| 3           | 45                  |
| 4           | 44                  |
| 5           | 45                  |
| 6           | 45                  |
| 7           | 44                  |
| 8           | 44                  |
| 9           | 45                  |
| 10          | 44                  |

2nd image-set:
| Number in sequence | Time (milliseconds) |
| ----------- | ------------------- |
| 1           | 193                 |
| 2           | 45                  |
| 3           | 85                  |
| 4           | 45                  |
| 5           | 45                  |
| 6           | 45                  |
| 7           | 45                  |
| 8           | 45                  |
| 9           | 45                  |
| 10          | 45                  |

3rd image-set:
| Number in sequence | Time (milliseconds) |
| ----------- | ------------------- |
| 1           | 196                 |
| 2           | 47                  |
| 3           | 91                  |
| 4           | 45                  |
| 5           | 49                  |
| 6           | 49                  |
| 7           | 46                  |
| 8           | 44                  |
| 9           | 45                  |
| 10          | 45                  |

4th image-set:
| Number in sequence | Time (milliseconds) |
| ----------- | ------------------- |
| 1           | 193                 |
| 2           | 86                  |
| 3           | 45                  |
| 4           | 44                  |
| 5           | 44                  |
| 6           | 44                  |
| 7           | 44                  |
| 8           | 44                  |
| 9           | 45                  |
| 10          | 44                  |

5th  image-set:
| Number in sequence | Time (milliseconds) |
| ----------- | ------------------- |
| 1           | 193                 |
| 2           | 44                  |
| 3           | 84                  |
| 4           | 44                  |
| 5           | 44                  |
| 6           | 44                  |
| 7           | 44                  |
| 8           | 45                  |
| 9           | 44                  |
| 10          | 44                  |

## Generated images
The link below leads to a Google Drive where all generated images are stored in folders. All folders also include a `.txt` file with settings used in the generation and what model was used. 

https://drive.google.com/drive/folders/1SHluCw8tG61whXpTKGgK37BQud0C8IcY?usp=sharing

## Sources for poster
Nicolas De Araújo Moreira, Rubem Vasconcelos, Yuri Carvalho Barbosa Silva, Tarcisio Ferreira Maciel, Ingrid Simoes, João César Moura Mota, Cerine Hamida, Rodrigo Zambrana Prado, Émilie Poisson Caillault, Modeste Kacou, et al. (2023) Convolutional Long-Short-Term Memory Networks (ConvLSTM) for Weather Prediction using Radar and Satellite Images. https://hal.science/hal-04079740v1/document

Statens Meteorologiska och Hydrologiska Institut (SMHI). (n.d., retrieved 19-09-2025). Moln - bindningssätt. https://www.smhi.se/kunskapsbanken/meteorologi/moln/moln---bildningssatt

EUMETSAT. (n.d., retrieved 28-09-2025) Meteosat series. https://www.eumetsat.int/our-satellites/meteosat-series 

A. Moreno, Meteorolog, (personal communication, 12-09-2025)
