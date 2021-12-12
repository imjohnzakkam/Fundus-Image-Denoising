import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from numpy import asarray
from unet import unet2
from tensorflow import keras

x_train = []
x_test = []

for i in os.listdir("./test_imgs_1/good"):
    sti = "./test_imgs_1/good/" + i
    image = Image.open(sti)
    data = asarray(image)
    data = np.array([cv2.resize(data, (256, 256))])
    x_train.append(data[0] / 255)

x_train = np.array(x_train)

for i in os.listdir("./test/bad"):
    sti = "./test/bad/" + i
    image = Image.open(sti)
    data = asarray(image)
    data = np.array([cv2.resize(data, (256, 256))])
    x_test.append(data[0] / 255)

x_test = np.array(x_test)

print(x_train.shape)

dimension = x_train.shape[1]

# x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

x_train = np.reshape(x_train, (len(x_train), dimension, dimension, 3))
x_test = np.reshape(x_test, (len(x_test), dimension, dimension, 3))

noise_factor = 0.01
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

n = 3
for i in range(n):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(5, 5)
    axes[0].set_title("True image")
    im0 = axes[0].imshow(x_test[i].reshape(dimension, dimension, -1), cmap="Reds")
    axes[1].set_title("Noisy image")
    im1 = axes[1].imshow(x_test_noisy[i].reshape(dimension, dimension, -1), cmap="Reds")
    plt.close(fig)

print(x_train_noisy.shape)

# autoencoder = unet2(dimension, dimension, 3)

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# validation_split = 0.8
# history = autoencoder.fit(
#     x_train_noisy,
#     x_train,
#     epochs=50,
#     batch_size=20,
#     shuffle=True,
#     validation_split=validation_split,
# )

# autoencoder.save('./saved_models/model_0')

autoencoder = keras.models.load_model('./saved_models/model_0')

# history.history.keys()

# train_loss = history.history["loss"]
# train_val_loss = history.history["val_loss"]
# epochs = range(1, len(train_loss) + 1)

# plt.figure(dpi=100)
# plt.plot(epochs, train_loss, label="Loss")
# plt.plot(epochs, train_val_loss, "o", label="Val loss")
# plt.title("Training and validation metrics")
# plt.legend()
# plt.savefig("history.png")
# plt.show()

all_denoised_images = autoencoder.predict(x_test_noisy)

test_loss = autoencoder.evaluate(x_test_noisy, x_test, batch_size=20)

def sharpen_img(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    return image

cnt = 1
n = 18
for i in range(n):
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(8, 2)
    axes[0].set_title("Noisy image")
    im0 = axes[0].imshow(x_test_noisy[i].reshape(dimension, dimension, -1))
    axes[1].set_title("Target image")
    im1 = axes[1].imshow(x_test[i].reshape(dimension, dimension, -1))
    axes[2].set_title("Denoised image")
    im2 = axes[2].imshow(all_denoised_images[i].reshape(dimension, dimension, -1))
    axes[3].set_title("Sharpened Image")
    final_img = all_denoised_images[i]
    final_img = sharpen_img(all_denoised_images[i]).reshape(dimension, dimension, -1)
    cnt += 1
    im3 = axes[3].imshow(final_img)
    plt.savefig(f"./comparision_images/comparison-{i}.png")
    plt.close(fig)
    # plt.imshow(final_img)
    # plt.savefig(f'./denoised/{cnt}_denoised.png')
