# Basic classification: Classify images of clothing
# 첫 번째 신경망 훈련하기: 기초적인 분류 문제

# This guide uses tf.keras, a high-level API to build and train models in TensorFlow.
# 여기에서는 텐서플로 모델을 만들고 훈련할 수 있는 고수준 API인 tf.keras를 사용합니다.

# TensorFlow and tf.keras
# tensorflow와 tf.keras를 임포트합니다
import tensorflow as tf
from tensorflow import keras

# Helper libraries
# 헬퍼(helper) 라이브러리를 임포트합니다
import numpy as np
import matplotlib.pyplot as plt

print("tensorflow version", tf.__version__)

# Import and load the Fashion MNIST data directly from TensorFlow:
# 패션 MNIST 데이터셋은 텐서플로에서 바로 임포트하여 적재할 수 있습니다:
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:
# 데이터셋에 클래스 이름이 들어있지 않기 때문에 나중에 이미지를 출력할 때 사용하기 위해 별도의 변수를 만들어 저장합니다:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

# Preprocess the data
# 데이터 전처리
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the training set and display the class name below each image.
# 훈련 세트에서 처음 25개 이미지와 그 아래 클래스 이름을 출력해 보죠. 데이터 포맷이 올바른지 확인하고 네트워크 구성과 훈련할 준비를 마칩니다.
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model
# 모델 구성
# Set up the layers
# 층 설정
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# 모델 훈련
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
# 정확도 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy [테스트 정확도]:', test_acc)

# Make predictions
# 예측 만들기
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(test_images)
# Let's take a look at the first prediction:
predictions[0]
# A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:
# 이 예측은 10개의 숫자 배열로 나타납니다. 이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)를 나타냅니다. 가장 높은 신뢰도를 가진 레이블을 찾아보죠:
np.argmax(predictions[0])

# So, the model is most confident that this image is an ankle boot, or class_names[9]. Examining the test label shows that this classification is correct:
test_labels[0]


# Graph this to look at the full set of 10 class predictions.
# 10개 클래스에 대한 예측을 모두 그래프로 표현해 보겠습니다:
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100 * np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Verify predictions
# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)


# plt.show()
print("i = 0")

# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()
print("i = 12")

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력합니다
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냅니다
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Use the trained model

# Grab an image from the test dataset.
# 테스트 세트에서 이미지 하나를 선택합니다
img = test_images[1]

print("test start - select img : ", "test_images[1]")
print("img.shape >> ", img.shape)

# Add the image to a batch where it's the only member.
# 이미지 하나만 사용할 때도 배치에 추가합니다
img = (np.expand_dims(img, 0))
print("np.expand_dims(img, 0) >> ", img.shape)
plt.figure()
plt.imshow(test_images[1])
plt.show()
# Now predict the correct label for this image:
# 이제 이 이미지의 예측을 만듭니다:
predictions_single = model.predict(img)
print("predictions_single", predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:
# model.predict는 2차원 넘파이 배열을 반환하므로 첫 번째 이미지의 예측을 선택합니다:
np.argmax(predictions_single[0])

# https://www.tensorflow.org/tutorials/keras/classification
print("FINISH :", np.argmax(predictions_single[0]))
print('FINISH offical tutorial and Answer is ', class_names[np.argmax(predictions_single[0])])
# FINISH. tutorial code.

# my add code >> select one image
# 나의 추가 코드 >> 이미지 한개 선택하여 해당 결과 값 확인
print()
print("START MY ADDED CODE===========================================================")
print()

i = 6
testimg2 = test_images[i]
print(testimg2.shape)
testimg = (np.expand_dims(test_images[i], 0))  # tf.keras 모델시
print(testimg.shape)

# show image window 이미지 보여주기 (기존이미지 사용)
print("testing show")
plt.figure()
plt.imshow(test_images[i])
plt.colorbar()
plt.grid(False)
plt.show()

# 해당 이미지에 대한 예측 (기존 이미지 사용) 그래프 보기
predictions_test = model.predict(testimg)
print("testing predictions_test :" , predictions_test)
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions_test[0], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(1, predictions_test[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

x = np.argmax(predictions_test[0])
print("x:", x)
print('.select image', str(i), 'The prediction Answer is ', class_names[x])
