code:

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y, verbose=0)
print(f"Accuracy: {acc*100:.2f}%")

predictions = model.predict(X, verbose=0)
print("Raw predictions:\n", predictions)
print("Rounded predictions:\n", np.round(predictions))

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
Output:
"C:\Users\Brindha T\Pictures\Screenshots\Screenshot 2025-08-10 194139.png"<img width="612" height="707" alt="Screenshot 2025-08-10 194139" src="https://github.com/user-attachments/assets/039e3c95-0d09-485c-abc6-a3103ce20d06" />
