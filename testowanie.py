from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# parameters
from sklearn.preprocessing import MinMaxScaler

model_type = Sequential()
epoch = 150
batch = 20
optimize = 'adam'
seed = 7
numpy.random.seed(seed)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# DATA - cleaning data - nans
df = pd.read_csv('water_potability.csv')
pd.set_option('display.max_columns', None)
df = df.dropna()
df.to_csv('water_potability2.csv')
# read cleaned data, skip first row with column names
dataset = loadtxt('water_potability2.csv', delimiter=',', skiprows=1)
# standardization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)
# split into input (X) and output (y) variables
X = dataset[:, 0:10]
y = dataset[:, 10]
# split for test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = KerasClassifier(build_fn=create_model, verbose=0)
batch_size = [10, 20, 30, 40]
epochs = [25, 50, 75, 85]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=0)
#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy * 100))




# plotting
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
#
# # make class predictions with the model
# predictions = (model.predict(X) > 0.5).astype(int)
#
# # summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
#
# print("sukces")
