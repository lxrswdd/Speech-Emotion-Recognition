# Speech Emotion Recognition Project

## General
- A model to classify the emotions of speeches
- Features were extracted by modified [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) library
- Preproccess the features.





## Feature Extraction
pyAudioAnalysis library is modified by the addition of functions in order to extract original features from .wav files and present them in 3D arrays.

### Modifications to `MidTermFeatures.py` 

#### `directory_feature_extraction_no_avg`
This function is able to extract features from a directory without averaging each file.

#### `multiple_directory_feature_extraction_no_avg`
This function is able to extract features from multiple directories without averaging each file. 

#### `directory_feature_extraction_no_avg_3D`
This function aims to extract audio features from a directory and turn a 3D array in terms of  (batch,step,features)

#### `multiple_directory_feature_extraction_no_avg_3D`
Multi-directories extraction for 3D array.
