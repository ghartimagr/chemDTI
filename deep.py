# Import packages
import os
import numpy as np
import pandas as pd
from math import floor
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')

print("Packages imported")

# Make scorer accuracy
score_acc = make_scorer(accuracy_score)

X_train = np.loadtxt("/work/ghartimagar/python_project_structure/data1.4/ionchannel_combined_1_4/atompair/X_train_undersampled.csv", delimiter=',')
y_train = np.loadtxt( "/work/ghartimagar/python_project_structure/data1.4/ionchannel_combined_1_4/atompair/y_train_undersampled.csv", delimiter=',')
X_test = np.loadtxt("/work/ghartimagar/python_project_structure/data1.4/ionchannel_combined_1_4/atompair/X_test_for_undersampled.csv", delimiter=',')
y_test = np.loadtxt("/work/ghartimagar/python_project_structure/data1.4/ionchannel_combined_1_4/atompair/y_test_for_undersampled.csv", delimiter=',')

print("Shapes:")
print("X_train:", X_train.shape[1])
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape) 

#Tune the layers
def tunelayers(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
    '''Inserting regularization layers in a neural network can help prevent overfitting. Two types:
    Batch normalization is placed after the first hidden layers.It normalizes the values passed to it for every batch.
    randomly drops a certain number of neurons in a layer. The dropped neurons are not used anymore. 
    The rate of how much percentage of neurons to drop is set in the dropout rate.'''
    optimizerL = ['SGD','Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
    optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
             'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
             'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
             'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu','elu', 'exponential', LeakyReLU,'relu']
    # res = isinstance(activation, str)
    # if res != True:
    #     activation = activationL[round(activation)]
    #     print ("is not a string")
    # else:
    #     activation = activation
    #     print ("is a string")
    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)

    def NeuralNetwork2():
        opt = Adam(learning_rate = learning_rate)
        nn = keras.Sequential()
        nn.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate))
        nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate))
        nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return nn
    
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=NeuralNetwork2, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train, scoring=score_acc, cv=kfold, fit_params={'callbacks':[es]}).mean()
    return score


params_nn2 ={
    'neurons': (10, X_train.shape[1]/2),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(200, 1200),
    'epochs':(20, 100),
    'layers1':(1,3),
    'layers2':(1,3),
    'normalization':(0,1),
    'dropout':(0,1),
    'dropout_rate':(0,0.3)
}
# Run Bayesian Optimization
nn_bo = BayesianOptimization(tunelayers, params_nn2, random_state=111)
nn_bo.maximize(init_points=25, n_iter=4)

params_nn_ = nn_bo.max['params']
learning_rate = params_nn_['learning_rate']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', 'LeakyReLU','relu']
print (activationL[int(params_nn_['activation'])])
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_['batch_size'] = round(params_nn_['batch_size'])
params_nn_['epochs'] = round(params_nn_['epochs'])
params_nn_['layers1'] = round(params_nn_['layers1'])
params_nn_['layers2'] = round(params_nn_['layers2'])
params_nn_['neurons'] = round(params_nn_['neurons'])
optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
             'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
             'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
             'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
params_nn_

print("Best parameters:", params_nn_, "\n")

# Get the best parameters from Bayesian Optimization
best_params = nn_bo.max['params']

# Extract the best hyperparameters
best_neurons = int(round(best_params['neurons']))
best_activation = int(round(best_params['activation']))
best_optimizer = int(round(best_params['optimizer']))
best_learning_rate = best_params['learning_rate']
best_batch_size = int(round(best_params['batch_size']))
best_epochs = int(round(best_params['epochs']))
best_layers1 = int(round(best_params['layers1']))
best_layers2 = int(round(best_params['layers2']))
best_normalization = best_params['normalization']
best_dropout = best_params['dropout']
best_dropout_rate = best_params['dropout_rate']

# Define the neural network with the best hyperparameters
def create_best_model():
    opt = Adam(learning_rate=best_learning_rate)
    model = keras.Sequential()
    model.add(Dense(best_neurons, input_dim=X_train.shape[1], activation='relu'))
    if best_normalization > 0.5:
        model.add(BatchNormalization())
    model.add(Dense(best_neurons, activation='relu'))
    if best_dropout > 0.5:
        model.add(Dropout(best_dropout_rate))
    model.add(Dense(best_neurons, activation='relu'))
    if best_dropout > 0.5:
        model.add(Dropout(best_dropout_rate))
    model.add(Dense(best_neurons, activation='relu'))
    if best_dropout > 0.5:
        model.add(Dropout(best_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Create the best model
best_model = create_best_model()

# Predictions on test set
y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))

# Precision, recall, and F1 score
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# ROC Curve and AUC-ROC Score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve and AUC-PR Score
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'AUC-PR = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred_binary)
print(f'Matthews Correlation Coefficient: {mcc:.4f}')




