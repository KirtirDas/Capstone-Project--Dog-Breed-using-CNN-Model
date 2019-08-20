# Capstone-Project--Dog-Breed-using-CNN-Model

# Table of Contents:
Installation
Project Overview
File Descriptions
Results
Licensing, Authors, and Acknowledgements

# Installation:
python 3.6
matplotlib
numpy
scikit-learn
keras
tqdm
ImageFile

# Project Overview:
The aim of the project was to develop an algorithm where the code will accept any user supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog’s breed.If a human is detected, it will provide an estimate of the dog breed that is most resembling and if neither is detected in the image, it provides output that indicates an error. Thus mainly our concern is to identify dog breeds from any image. The data set imported from sklearn provided in the workspace consists of 8351 dog images in 133 different categories. The project has been successfully executed in Udacity workspace(GPU-enabled) with Tensorflow in Keras. Even though prediction of dog breed is the subject of this project, but the technology can be applied to identifying letters, faces, handwritten recognition and tumors. 

# File Descriptions
Below are main foleders/files for this project:

haarcascades
haarcascade_frontalface_alt.xml: a pre-trained face detector provided by OpenCV
bottleneck_features
DogVGG19Data.npz: pre-computed the bottleneck features for VGG-19 using dog image data including training, validation, and test
saved_models
VGG19_model.json: model architecture saved in a json file
weights.best.VGG19.hdf5: saved model weights with best validation loss
dog_app.html: a jupyter notebook used to build and train the dog breeds classification model
extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image
images: a few images to test the model manually

# Results
The accuracy obtained was nearly 84% using CNN with transfer learning. Among the different bottleneck features, Xception performed best. This result is quite impressive as compared to CNN from scratch with accuracy with 6%.
For any other image uploaded, an estimate of the most resembling dog breed will be given and if neither is detected in the image, it provides error output.

Project files can be found in this github repo: https://github.com/swang13/dog-breeds-classification
Further briefing can be found in this blog: https://medium.com/@das.kirtirjasaswini/dog-breed-classifier-using-cnn-model-data-science-capstone-project-8207a43321eb

# Licensing, Authors, Acknowledgements
Credits must be given to Udacity for the starter codes and data images used by this project.
References:
https://medium.com/@mishaallakhani/detecting-skin-cancer-using-deep-learning-82deba830328 
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/


© 2019 GitHub, Inc.

