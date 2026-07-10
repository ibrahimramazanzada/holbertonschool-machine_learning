#!/usr/bin/env python3
'''Input model implementation with keras'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    '''Trains a model using mini-batch gradient descent'''

    callbacks = []
    if validation_data is not None and early_stopping:
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)

    if validation_data is not None and learning_rate_decay:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)
        learning_rate_decay = K.callbacks.LearningRateScheduler(scheduler,
                                                                verbose=1)
        callbacks.append(learning_rate_decay)

    if validation_data is not None and save_best and filepath:
        save_best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_loss',
                                                save_best_only=True)
        callbacks.append(save_best)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)
    return history
