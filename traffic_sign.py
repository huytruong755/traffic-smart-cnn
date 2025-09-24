import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dense, Flatten, Dropout
import os

data = []
labels = []
classes = 43
cur_path = os.getcwd()

print("Đang tải ảnh lên...")
for i in range(classes):
    path = os.path.join(cur_path, 'Training', str(i))
    if not os.path.exists(path):
        print(f"Warning: Folder {path} does not exist.")
        continue
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(os.path.join(path, a)).convert('RGB')
            image = image.resize((128, 128))  # Tăng kích thước ảnh
            image = np.array(image, dtype=np.float32)
            data.append(image)
            labels.append(i)
print("Đang chuyển đổi dữ liệu thành mảng NumPy...")
data = np.array(data, dtype=np.float32)
labels = np.array(labels)
print(data.shape, labels.shape)
data = data / 255.0
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=43)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
print("Đã lưu X_test.npy và y_test.npy thành công!")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=X_train.shape[1:], padding='same'))
model.add(Conv2D(32, (5,5), activation='relu', strides=2, padding='same'))  # Thay pooling bằng stride=2
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', strides=2, padding='same'))  # Thay pooling bằng stride=2
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.add(Dense(128*128*3, activation='relu'))
model.add(Reshape((128,128,3)))
model.add(UpSampling2D((2,2)))
model.add(Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Loss: {loss * 100:.4f}%")
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
