from ultil import *
import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2,l1
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Embedding,Bidirectional,Masking,LSTM,BatchNormalization,Input
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHead
import os

steps = 50 #定义timestep; define timestep

#数据组地址定义; Define the data directory
source_dir = '/mnt/lxr/datasets/Fujitsu/binary'
X_train_dir = os.path.join(source_dir,'X_training.csv')
y_train_dir = os.path.join(source_dir,'y_training.csv')
X_test_dir = os.path.join(source_dir,'X_testing.csv')
y_test_dir = os.path.join(source_dir,'y_testing.csv')

#读取数据; Read the data
X_train,y_train,X_test,y_test,num_classes = load_dataset(X_train_dir,y_train_dir,X_test_dir,y_test_dir,steps,0,136,True)

#数据处理; Feature engineering
X_train,y_train = binary_smote(X_train,y_train) #对训练集进行SMOTE
X_train = masked_normalization(X_train) #对数据集进行统一化
X_test = masked_normalization(X_test) #对数据集进行统一化
X_train = PCA_1(X_train,dim) #对数据集进行降维
X_test = PCA_1(X_test,dim) #对数据集进行降维

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=666) #对训练集进行训练集验证集分割



def opt_select(optimizer, learning_rate):

    """
    为模型选择优化器
    Selection of optimizer
    
    :param optimizer:优化器名字
    :param learning_rate: 学习率
    :return:
    """
    if optimizer == 'Adam':
        adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        return adamopt

    elif optimizer == 'SGD':

        SGDopt = tf.keras.optimizers.SGD(lr=learning_rate)
        return SGDopt

    elif optimizer == 'RMS':

        RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
        return RMSopt

    else:
        print('undefined optimizer')


def residual_attention_model(X_train, y_train, X_val, y_val, X_test, num_classes, dropout=0.2, batch_size=68,
                    learning_rate=0.0001, epochs=20, optimizer='Adam'):
    """residual attention 模型
       residual attention model
    """
    lstm_unit = 16

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Masking(mask_value=0.0)(inputs)
    x2 = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                          attention_activation='sigmoid')(x)
    x = x + x2
    x = Bidirectional(LSTM(lstm_unit, dropout=dropout, return_sequences=True))(x)
    x = Flatten()(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    print(model.summary())

    optimizer = opt_select(optimizer, learning_rate)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
                 ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')]

    hist = model.fit(X_train,
                     y_train,
                     shuffle=False,
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=callbacks,
                     verbose=0,
                     validation_data=(X_val, y_val))

    model.load_weights(filepath = '.mdl_wts.hdf5')
    model.save('/mnt/lxr/SER/paper/fiji_binary.h5')

    yhat = model.predict(X_test)

    return hist, yhat


def Multiplcative_self_attention(X_train, y_train, X_val, y_val, X_test, num_classes, dropout=0.2, batch_size=68,
                         learning_rate=0.0001, epochs=20, optimizer='Adam'):
    """多层self_attention;
    Multiplicative self_attention model
    """
    lstm_unit = 256

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Bidirectional(LSTM(lstm_unit, dropout=dropout,return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_unit, dropout=dropout,return_sequences=True)))

    model.add(SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation='sigmoid',
        kernel_regularizer=keras.regularizers.l2(1e-2),
        use_attention_bias=False,
        name='Attention',
    ))

    model.add(keras.layers.Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    opt = opt_select(optimizer, learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min'),
                 ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        verbose=0
                        )

    model.load_weights(filepath='.mdl_wts.hdf5')
    model.save('/mnt/lxr/SER/paper/fiji_binary.h5')

    yhat = model.predict(X_test)

    return history, yhat


def MultiHead_self_attention(X_train, y_train, X_val, y_val, X_test, num_classes, dropout=0.5, batch_size=68,
                         learning_rate=0.0001, epochs=20, optimizer='Adam'):
    """Multi-Head attention 模型"""


    lstm_unit = 256

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MultiHead(Bidirectional(LSTM(units=lstm_unit, dropout=dropout)), layer_num=10, name='Multi-LSTMs'))

    model.add(SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation='sigmoid',
        kernel_regularizer=keras.regularizers.l2(1e-2),
        use_attention_bias=False,
        name='Attention',
    ))

    model.add(keras.layers.Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    opt = opt_select(optimizer, learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min'),
                 ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')]

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        verbose=0
                        )

    model.load_weights(filepath='.mdl_wts.hdf5')
    model.save('/mnt/lxr/SER/paper/fiji_binary.h5')

    yhat = model.predict(X_test)

    return history, yhat

#模型参数定义; Define the hyperparameters
num_classes = num_classes
batch_size = 256
epochs = 20
learning_rate = 0.001

#训练模型; Model training
history,y_pred = MultiHead_self_attention(X_train, y_train, X_val, y_val,X_test,num_classes=8,dropout=0.2, batch_size=64, learning_rate=learning_rate,epochs=epochs,optimizer='Adam')
#训练结果 # plot the report and confusion matrix and acc loss curves
plots(history,y_test,y_pred)
