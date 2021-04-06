from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.decomposition import PCA
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioTrainTest as aT
from sklearn.utils import class_weight
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def load_dataset(X_train_directory,y_train_directory,X_test_directory,y_test_directory,window_num,trunc_start,truc_end,truncation = False):
    """
    该函数能够读取已经抽取好的音频特征，并对其进行切割。
    This function reads extracted features from CSV file and truncate the features.
    
    :param X_train_directory: 训练组地址
    :param y_train_directory: 测试族标签地址
    :param X_test_directory: 测试组地址
    :param y_test_directory: 测试组标签地址
    :param window_num: timestep 数量
    :param trunc_start: 特征剪裁起始点，最低值0
    :param truc_end: 特征剪裁终点，最高值136
    :param truncation: 默认False,不进行特征剪裁，保留0-136特征
    :return: 训练组特征
    """

    X_train = np.loadtxt(X_train_directory, delimiter=',')
    X_train = X_train.reshape(X_train.shape[0],window_num,136)
    y_train = np.loadtxt(y_train_directory, delimiter=',')

    X_test = np.loadtxt(X_test_directory, delimiter=',')
    X_test = X_test.reshape(X_test.shape[0],window_num,136)
    y_test = np.loadtxt(y_test_directory, delimiter=',')
    
    if truncation:
        X_train = X_train[:,:,trunc_start:truc_end]
        X_test = X_test[:,:,trunc_start:truc_end]
    
    num_classes = y_train.shape[1]
    
    print('load succeed')
    print('There are ',num_classes,'classes')
    print('shape of X_train: ', X_train.shape)
    print('shape of X_test: ', X_test.shape)
    print('shape of y_train: ', y_train.shape)
    print('shape of y_test: ', y_test.shape)

    return X_train,y_train,X_test,y_test,num_classes

def extract_no_avg_3Dfeatures(path,mid_window=0.1,mid_step=0.1,short_window = 0.05,short_step=0.025,steps = 50):
    """
    对多目录进去音频特征抽取。
    :param path: 语音文件夹，['.../happy','.../angry']
    :param mid_window: 中尺寸窗口大小
    :param mid_step: 窗口移动值
    :param short_window: 小尺寸窗口大小
    :param short_step: 小尺寸窗口移动值
    :param steps: timestep. mid_windo*steps = 声音分析的时常
    :return: 特征组，标签，类名
    """

    features, class_names, file_names = aF.multiple_directory_3Dfeature_extraction_no_avg(path,mid_step,mid_step,short_window,short_step,steps)
    feature_matrix, labels = aT.features_to_matrix(features)
    return feature_matrix,labels,class_names


def masked_normalization(X):
    """
    对单时序数据矩阵进行统一化。 fit 自身，然后transform自身。
    :param X: X_train或者X_test
    :return:
    """

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # This function normalizes datasets without the including of 0's
    nz = np.any(X, -1)
    sc = StandardScaler().fit(X[nz])
    X[nz] = sc.transform(X[nz])

    return X

def masked_normalization2(X,Y,scaler):
    """
    对时序数据进行标准化或者统一化。X可以是X_train, Y可是X_test.
    :param X: 数据fit源和目标
    :param Y: 数据目标
    :param scaler: 选择统一化模式，minmaxscaler 或者 standardscaler.
    :return:
    """

    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    # This function normalizes datasets without the including of 0's
    X, Y = np.copy(X), np.copy(Y)
    nz = np.any(X, -1)
    
    if scaler == 'minmax':
        sc = MinMaxScaler().fit(X[nz])
    else:
        sc = StandardScaler().fit(X[nz])
        
    X[nz] = sc.transform(X[nz])
    Y[np.any(Y, -1)] = sc.transform(Y[np.any(Y, -1)])
    
    return X,Y


def extract_and_process(directory):
    """
    directory : A list of directory
    """
    features,y,_ = extract_no_avg_3Dfeatures(directory)
    X = features
    
    # masked normalization
    scaled_X = masked_normalization(X)
    
    # PCA reduction
    reduced_dimension = 40
    scaled_X = np.reshape(scaled_X,(-1,scaled_X.shape[2]))
    pca = PCA(n_components = reduced_dimension)
    pca.fit(scaled_X)
    scaled_X = pca.transform(scaled_X)
    X  = np.reshape(scaled_X,(X.shape[0],X.shape[1],reduced_dimension))
    return X,y



def PCA_1(X,dim):
    """
    对数据组进行PCA降维
    :param X: 数据
    :param dim: 目标维度
    :return:
    """
    reduced_dimension = dim

    scaled_X = np.reshape(X,(-1,X.shape[2]))
    
    pca = PCA(n_components = reduced_dimension)
    pca.fit(scaled_X)
    scaled_X = pca.transform(scaled_X)
    
    X  = np.reshape(scaled_X,(X.shape[0],X.shape[1],reduced_dimension))
    return X




def binary_smote(X_train,y_train):
    # 对多分类模型进行SMOTE。
    sm = SMOTE()
    X_train_original = X_train
    y_train_original = y_train
    
    X_train = np.reshape(X_train,(X_train.shape[0],-1))
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('y shape after smote fit',y_train.shape)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train_original.shape[1],X_train_original.shape[2]))
    y_train = tf.keras.utils.to_categorical(y_train, y_train_original.shape[1],dtype='float32')
    
    return X_train,y_train

def smote(X_train,y_train):
    # 对2分类模型进行SMOTE。

    sm = SMOTE()
    X_train_original = X_train
    y_train_original = y_train
    
    X_train = np.reshape(X_train,(X_train.shape[0],-1))
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print('y shape after smote fit',y_train.shape)
    X_train = np.reshape(X_train,(X_train.shape[0],X_train_original.shape[1],X_train_original.shape[2]))
    
    return X_train,y_train



def plots(history,y_test,y_pred):
    """
    1.对模型训练记录绘acc和loss图
    2.绘制混淆矩阵
    3.打印分类报告
    :param history: model.fit()的history数据
    :param y_test:测试组标签
    :param y_pred: model.predict()的数据
    :return:
    """

    line_length = 0
    fig = plt.figure(figsize=(14, 34))

    ax = fig.add_subplot(10,2,1)
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    # plt.hlines(0.75,300,line_length,'g')

    title = 'acc'
    ax.title.set_text(title)

    ax = fig.add_subplot(10,2,3)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    # plt.hlines(0.6,600,line_length,'g')

    title = 'loss'
    ax.title.set_text(title)

    min_loss = min(history.history['val_loss'])
    min_index = history.history['val_loss'].index(min_loss)
    highest_acc = history.history['val_accuracy'][min_index]
    print('min loss ',min_loss)
    print('highest acc',highest_acc)

    mat = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
    plot_confusion_matrix(mat)

    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
