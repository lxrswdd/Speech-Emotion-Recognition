import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Embedding,Bidirectional,Masking,LSTM

def LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005, epochs=20,
               dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout))

    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def bi_LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                  epochs=20, dropout=0.2, recurrent_dropout=0.2):

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def duo_LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                   epochs=20, dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
    model.add(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def duo_bi_LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                      epochs=20, dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def trio_bi_LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                       epochs=20, dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def quad_bi_LSTM_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                       epochs=20, dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.90):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat


def trio_bi_LSTM_Dnn_model(X_train, y_train, X_test, y_test, num_classes, batch_size=68, units=128, learning_rate=0.005,
                           epochs=20, dropout=0.2, recurrent_dropout=0.2):
    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > 0.9):
                print("\nReached 90% accuracy so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(
        LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    adamopt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    RMSopt = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
    SGDopt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.1, nesterov=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adamopt,
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callbacks])

    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)

    yhat = model.predict(X_test)

    return history, yhat