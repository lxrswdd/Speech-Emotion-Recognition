# Speech-Emotion-Recognition
This project involves in speech features extraction and training.


# Feature Extraction
A self-modified pyAudioAnalysis package is applied to extract audio features from the .wav files.

## Additional Functions created on the base of functions in `MidTermFeatures.py` from pyAudioAnalysis

### `directory_feature_extraction_no_avg`
This function is able to extract features from a directory without averaging each file.

### `multiple_directory_feature_extraction_no_avg`
This function is able to extract features from multiple directories without averaging each file. 

### `directory_feature_extraction_no_avg_3D`
This function aims to extract audio features from a directory and turn a 3D array in terms of  (batch,step,features)

### `multiple_directory_feature_extraction_no_avg_3D`
Multi-directories extraction for 3D array.
