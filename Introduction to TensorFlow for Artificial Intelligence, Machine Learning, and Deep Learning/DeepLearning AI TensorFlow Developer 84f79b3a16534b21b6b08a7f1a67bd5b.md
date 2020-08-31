# DeepLearning.AI TensorFlow Developer

Welcome to Deep Learning with Tensorflow version

Before we begin we need to install Tensorflow as follows

> !pip install tensorflow==2.0.0

# Week - 1

## A primer in machine learning

In traditional programming the rules and data gets in and the answers are came out whereas in machine learning the answers and data gets in and the rules gets out.

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture.jpg)

## The 'Hello World' of neural network

```python
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
```

In keras, you use the word dense to define a layer of connected neurons. So here thers is only one Dense here so only one layer and units =1 means only one neuron is here and input_shape=[1] shows that there is only one input like single input called X for the output y. And each and every layers are in sequence hence we have used the word called Sequential

```python
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
# here the optimizer is stochastic gradient descent = 'sgd'
```

There are two function we should be aware of is the optimizers and the loss function. The idea is like this that the neural network doesn't have any idea about the relationship between the x and y.

This is like suppose the actual relationship is like y=2x-1. but the neural network at the beginning guess it like y=10x-3. and then check this with the data which we provide to them. The loss function measures how good or how bad our guess was and the optimizer which figures out the next guess. And the logic is that each guess should be better than before

The term 'Convergence' is used when the accuracy reaches the 100%.Once the model is created we need to provide them the data like x and y for it

```python
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
# here the optimizer is stochastic gradient descent = 'sgd'

xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
ys=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)
```

The np.array is used to make the list as an array using python library called as numpy and making the datatype as 'float' using the dtype function. And the training happens using the fit function like below and parameters for the fit function is the x value , y value and other parameters like epochs(which is nothing but a no of iterations)

'predict' method gives the results from the model

```python
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_eror')
# here the optimizer is stochastic gradient descent = 'sgd'

xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
ys=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)

model.fit(xs,ys, epochs=500)

# Now we are going to use predict method to find the value for particular 
# x values
print(model.predict([10.0]))
```

# Week - 2

## Introduction to Computer Vision

Computer vision is the field of having a computer understand and label what is present in an image. Now in this discussion we are going to understand how the machine understand the neural network learn from the images and understand that this images is shoe or this image is shirt like wise.

Here we are going to use the dataset called Fashion MNIST where we have each images as 28X28 and there are 70,000 images of different varieties of the clothings. It has 10 Categories.

## Lets start coding from beginning = "loading the training data"

```python
# Here tensorflow have made the Fashion MNIST data as a dataset

### Splitting the training and testing dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
```

Now let us define a new neural network naming **Convolutional Neural Network(CNN)** architecture

```python
model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nan.softmax)
])
```

The important thing we need to understand is that the first and the last layer where the ***input_shape*** here is the image size(Fashion MNIST images are of (28X28) size) and the last layer denotes Dense(10) denotes the number of target variable classes (here we are having the target classes like 10 ). 

The interesting thing happen in the middle layer also called as the hidden layer and it has 128 neurons as the units place has 128 neuron

## Walk through a Notebook for Computer Vision

Neural network actually works better in the normalized data like suppose u have a data like 1,3,4,5,8,6,10 then we will divide the data with maximum value in the set like dividing it by 10

```python
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# Normalising the training and testing images so neural network works well
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nan.softmax)
])

model.compile(optimizer = tf.train.AdamOptimizer(),
							loss = 'sparse_categorical_crossentropy')
model.fit(train_images,train_labels,epochs=5)

# finally we need to find how good ur model with test data from below code

model.evaluate(test_images,test_labels)

```

## Using Callbacks to control training

If suppose we are using 10 epochs for training and then u found that after 6 epochs the loss of the neural network is increasing and not decreasing after that so tensorflow do support the callback function which is implemented in the below code

```python
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28,28)),
		# converts the 2-D array into single 1-D array
		keras.layers.Dense(128, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nan.softmax)
])

model.compile(optimizer = tf.train.AdamOptimizer(),
							loss = 'sparse_categorical_crossentropy')
## We will define the call backs in the fit function
# before that let us define the class for that call backs
class myCallback(tf.keras.callbacks.Callback):
		def on_epoch_end(self,epoch,logs={}):
				if(logs.get('loss')<0.4):
						print("\nLoss is low so cancelling training!")
						self.model.stop_training = True
# initiating the myCallback function
callbacks = myCallback()
model.fit(train_images,train_labels,epochs=5,callbacks=[callbacks])

# finally we need to find how good ur model with test data from below code

model.evaluate(test_images,test_labels)
```

The important thing is that the callback happens after completing the epoch like in third epoch if the loss was approaching the 0.4 then before 4th epoch the training ends and here we can customize based on the epoch values in the class function

# Week - 3

## Convolutional and Pooling

Like in previous video we have discussed that how we have used a matrix having the shape like (70000,28x28) matrix and used DNN to predict the labels whereas here when we come to have so many shapes and sizes of images in RGB then it may fail or we can say it is advisable to use CNN Architecture.

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%201.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%201.jpg)

Here the each and every images are taking the pixels and then multiplied with the filter and create a new pixel values

```python
# Convolutional Neural Network looks like this
model = tf.keras.models.Sequential([
				tf.keras.layers.Conv2D(64,(3,3),activation='relu',
																				input_shape=(28,28,1)),
				tf.keras.layers.MaxPooling2D(2,2),
				tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
				tf.keras.layers.MaxPooling2D(2,2),
				tf.keras.layers.Flatten(),
				tf.Dense(128,activation='relu'),
				tf.Dense(1,activation='relu')
])
# it will inspect the model 
model.summary()
```

The model summary will something looks like this we will discuss this deeply in below 

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%202.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%202.jpg)

Here to notice is that here input shape of the Conv2D is (28,28,1) and the output shape is (26,26,64). The reason behind is that we are using (3,3) filter which means that suppose we have table of (5,5) image then the filter is applied to the corresponding (3,3) matrix as a subset of the conv2d so its simply like this

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%203.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%203.jpg)

The top and bottom will be rejected so that only 26 pixels are used up in 28 and in left and right will be rejected by 1 pixel each so that only 26 pixels are used up in 28. So the filter shape will surely implement the output shape of the cnn model

Next,let us discuss the max_pooling2d where we use (2,2) so it is something like it will be the (2,2) matrix will take one element or we can say something like if we have (16,16) matrix and used (2,2) then the output shape will be something like (8,8)

 

```python
**import tensorflow as tf

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('acc')>0.998):
        print("\n 99% acc reached")
        self.model.stop_training = True

def preprocess_images(image_set):
  image_set = image_set.reshape(-1, 28, 28, 1)
  image_set = image_set / 255.0
  return image_set

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = preprocess_images(training_images)
test_images = preprocess_images(test_images)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_images,
    training_labels,
    batch_size=64,
    epochs=20,
    callbacks=[CustomCallbacks()]
)**
```

# Week - 4

## UnderStanding ImageGenerator

The image generator is something like if we don't have same image dimensions and aspect ratio's then it would be difficult to load the with some modifications each and every time. This is how the image generators will work and the flow chart is like below

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%204.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%204.jpg)

So we will use the Image directory to training then the y_labels will be horses and humans and the x_labels will be the corresponding images inside the sub-directory of each.

```python
# importing the image datagenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Instantiate the imagedatagenerator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_dir, # path goes here
									target_size = (300,300),
									batch_size=128,
									class_mode='binary') 
```

## Defining a ConvNet to use complex images

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%205.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%205.jpg)

Here the input_shape is changed rather than using the fashion_input=(28,28,1) and then we have changed the neuron size from 32 to 64 in the second hidden layer and then at the last we have used the dense output as 1 and activation as sigmoid. So why we are using these at the last output layer because 1 denotes the binary classification as 1 and 0 where 1 -True and 0-False and we are using sigmoid activation because it values are between 0 and 1 

> It is better to use 1 and 'sigmoid' for the binary classification and n_labels and 'softmax' for the multi class classification

The model.summary will something like this.....

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%206.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%206.jpg)

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
```

Here we are using binary_crossentropy rather than categorical cross entropy in the previous compilation since we are using binary classification(2-label classification). and we are using rmsprop optimizer with lr(learning rate of the optimizer ) 

![DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%207.jpg](DeepLearning%20AI%20TensorFlow%20Developer%2084f79b3a16534b21b6b08a7f1a67bd5b/Capture%207.jpg)

here we are using the train_generator inside the fit method as we already discussed earlier and steps_per_epoch=8 is the which is nothing but the no of images inside the directory // no of batch size, and validation_data = testing datagenerator and the validation steps = no of testing examples// no of batch_size and verbose =2 (shows no output)

## Walking through developing a ConvNet

```python
# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class CustomCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>DESIRED_ACCURACY):
                print("\n 99% acc reached")
                self.model.stop_training = True

    callbacks = CustomCallbacks()

    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=64,
        class_mode='binary'
    )

    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
    train_generator,
    epochs=20,
    callbacks=[callbacks]
    )
    # model fitting
    return history.history['acc'][-1]

# calling the method
train_happy_sad_model()
```