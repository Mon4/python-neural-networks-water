from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# parametry
epoch = 100
batch = 25
model_type = Sequential()
optimize = 'adam'

# czyszczenie danych - nans; standaryzacja
df = pd.read_csv('water_potability.csv')
df = df.dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

df.to_csv('water_potability2.csv')


# ponowne wczytanie, tym razem oczyszczonych danych, opuszczamy pierwszy wiersz, gdzie są nazwy kolumn
dataset = loadtxt('water_potability2.csv', delimiter=',', skiprows=1)

X = scaled[:, 0:9]
y = scaled[:, 9]

# stworzenie zestawów danych trenujących i testowych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# budowanie modelu
model = model_type
model.add(Dense(15, input_dim=9, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model
model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=0)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))


# wykresy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
