import numpy as np
import tensorflow as tf
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import ShortTermFeatures

Training_path=['']
Testing_path=['']

def single_file_extraction(file_directory):
    from pyAudioAnalysis import audioBasicIO
    import matplotlib.pyplot as plt
    [Fs, x] = audioBasicIO.read_audio_file(file_directory)
    Features = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    print('Single file short term feature extraction: ')
    print(Features)
    plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
    plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

def multidirectory_features_extraction(path):
    features = aT.extract_features(path, 1.0, 1.0, 0.02, 0.02)
    # print(features)
    # print(len(features))
    [X, y] = aT.features_to_matrix(features)
    print('dimension of Voice features:\n', X.shape)
    print('---------------------------------------------------------------------')
    print('Class labels: \n', y)
    return X, y

print('-----------------------------------------------------')
print('Extracting features from training data')
print('-----------------------------------------------------')
X_train,y_train = multidirectory_features_extraction(Training_path)
print('-----------------------------------------------------')
print('Extracting features from testing data')
print('-----------------------------------------------------')
X_test,y_test = multidirectory_features_extraction(Testing_path)

#Parameters
num_classes = 8

# OneHotEncode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes,dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes,dtype='float32')

print('--------------------------------------------------')
print('Shape of training set',X_train.shape)
print('There are {} features and {} sets of data'.format(X_train.shape[1],X_train.shape[0]))
print('--------------------------------------------------')
print('Saving data')

# Save the datasets to local.
np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")