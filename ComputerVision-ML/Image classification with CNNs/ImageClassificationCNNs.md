Welcome to the Image Classification Workshop!

The objective is to get a first hands-on with neural networs. It is also a good opportunity to acquire some Keras skills. Keras is a very widespread deep learning tool that is based on TensorFlow.

The task that you're given is to classify the test split of CIFAR10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html), after training a neural network with its train set.

You are given a toy network on which you can rely, and build a more optimized one. Bear in mind that there are some intentional mistakes in the code. You must also experiment with all the hyperparameters such as activation functions, loss function etc., as they are not the optimal ones. Let's begin!

First, we need to import some things:

keras is the DL library,

numpy is used for arrays,

PIL is used to handle images,

IPython.display is used in order to be able to show images on the notebook,

cv2 (openCV) is a computer vision library used for basic cv functions


```python
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from PIL import Image
from IPython.display import display
import cv2
```

CIFAR10 is already included in Keras, so we just load it in two tuples, one for the train split, one for the test split. x indicates images and y labels (10 classes, so labels can be in the range 0,...,9)


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print("Training dataset images shape: " +str(x_train.shape))
print("Training dataset labels shape: " +str(y_train.shape))
print()
print("Test dataset images shape: " +str(x_test.shape))
print("Test dataset labels shape: " +str(y_test.shape))
```

Let's test that our dataset is successfully loaded and visualize some images.


```python
test_image = x_train[4342]
# print(test_image[:,:,0])
test_image = cv2.resize(test_image, (200, 200))
test_image = Image.fromarray(test_image)
display(test_image)
print(y_train[4342])
```


    
![png](output_5_0.png)
    


    [2]
    


```python
num_classes = 10
print("Label of the first training example BEFORE one-hot encoding:" +str(y_train[0]))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("Label of the first training example AFTER one-hot encoding:" +str(y_train[0]))
```

    Label of the first training example BEFORE one-hot encoding:[6]
    Label of the first training example AFTER one-hot encoding:[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    

In the following function we finally define our CNN model! It consists of a convolutional layer, a maxpooling one, another convolutional and a fully connected one (Dense). As Dense layers can only receive 1D inputs, we use Flatten() to unroll the 3D activation volume of the 2nd convolutional layer to a 1D vector.

- For activation function some common choises are: a) 'sigmoid', b) 'softmax', c) 'tanh', d) 'relu', etc.

- For loss function some common choises are: a) CategoricalCrossentropy(), b) MeanSquaredError(reduction='sum_over_batch_size'), etc.

- For optimizers some common choises are: a) Adam(), b) SGD(), RMSprop(), etc.

- You can also play with the number of filters in each layer starting from something like: 64-> 128 -> 256, if we are using 3 Conv2D layers.


```python
def define_Model():

  initializer = keras.initializers.GlorotNormal()

  model = Sequential()

  model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32, 32, 3), padding='same', kernel_initializer=initializer))
  # model.add(Dropout(0.3))
  # model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer=initializer))
  # model.add(Dropout(0.3))
  # model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer=initializer))
  # model.add(Dropout(0.3))
  # model.add(BatchNormalization())

  model.add(Flatten())

  model.add(Dense(100, activation='relu', kernel_initializer=initializer))
  model.add(Dense(10, activation='softmax', kernel_initializer=initializer))

  model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

  return model
```


```python
 model = define_Model()
 model.summary()
```

The below functions can be used in order to apply some standard normilisation preprocessing and augmentation techniques.


```python
def preprocess(x, y):
  '''print(x_train[0,10,10])
 

  red_mean = np.mean(x_train[:,:,:,0])
  green_mean = np.mean(x_train[:,:,:,1])
  blue_mean = np.mean(x_train[:,:,:,2])

  x_train[:,:,:,0] = x_train[:,:,:,0] - red_mean
  x_train[:,:,:,1] = x_train[:,:,:,1] - green_mean
  x_train[:,:,:,2] = x_train[:,:,:,2] - blue_mean'''

  x = x.astype('float64')

  x = x/255

  return x, y

def augmentation(x_train, y_train):

  x_train_augmented = x_train
  x_train2 = x_train[:,:,::-1]

  y_train = np.append(y_train, y_train, axis=0)

  x_train = np.append(x_train_augmented, x_train2, axis=0)
 
  return x_train, y_train
```


```python
epochs = 10     # Number of epochs hyperperameter (how many times the whole dataset will be seen by the network). This parameter can be tweaked.
batch_size = 32 #Number of samples in each batch hyperperameter. This parameter can be tweaked.

# x_train, y_train = preprocess(x_train, y_train) # This line can be uncommented in order to apply some normilisation to the training data.
# x_test, y_test = preprocess(x_test, y_test)     # This line can be uncommented in order to apply some normilisation to the test data.

# x_train, y_train = augmentation(x_train, y_train)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) # fit() starts the backpropagation algorithm.
```

Thankn you for attending our workshop. For any questions/clarifications/comments please contact at emmanoup@csd.auth.gr.


```python

```
