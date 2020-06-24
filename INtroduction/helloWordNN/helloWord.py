from keras.layers import Dense
from keras.models import Sequential
import numpy as np

model = Sequential([Dense(units=1,input_shape=[1])])
model.compile(optimizer="sgd",loss="mean_squared_error")

X  = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(X, Y, epochs=500)
print(model.predict([10.0]))