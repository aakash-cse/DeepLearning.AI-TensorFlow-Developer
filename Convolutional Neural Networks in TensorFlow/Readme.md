# Convolutional Neural Networks in TensorFlow

# **Week - 1**

## Training with cats vs dogs dataset

We have gone through the fashion datasets where images were small and focused on the subject like dress alone....

```python
# importing the image datagenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Instantiate the imagedatagenerator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  train_dir, # path goes here
									target_size = (150,150),
									batch_size=20,
									class_mode='binary') 

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
                  test_dir, # path goes here
									target_size = (150,150),
									batch_size=20,
									class_mode='binary') 

model = tf.keras.models.Sequential([
									tf.keras.layers.Conv2D(16,(3,3),activation='relu',
																					input_shape=(150,150,3)),
									tf.keras.layers.MaxPooling2D(2,2),
									tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
									tf.keras.layers.MaxPooling2D(2,2),
									tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
									tf.keras.layers.MaxPooling2D(2,2),
									tf.keras.layers.Flatten(),
									tf.keras.layers.Dense(512,activation='relu'),
									tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),metrics=['acc'],
							loss='binary_crossentropy')

history = model.fit_generator(train_generator,steps_per_epoch=100,
															epochs=15,validation_data=validation_generator,
															validation_steps=50,verbose=2)
```

## Looking at accuracy and loss

```python
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

import matplotlib.pyplot as plt
plt.plot(epochs,acc)
plt.plot(epochs,val_acc)
plt.title("Training and validation Accuracy")
plt.figure()

plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title("Training and validation Loss")
plt.figure()
```

## Week -2

The image augumentation and data augumentaiton is one of the most widely used tools in deep learning to increase your dataset size and make your neural network perform better. Here important thing to be noted is that the data is not overriding here rather than we are just flowing it from the directory and then we are manipulating it and fed into the neural network which makes us to check for the various data augumentation techniques

There is a high change to make our cnn model to be overfit which is nothing but high bias and low variance. It oftenly happens when the data is smallar in size and as a result we can have some mistakes in our classification

The imagedatagenerator which we had actually used will do the needful for the image augumentation by supplying the various parameters to the method

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled.png)

This week is all about the image augumentation and this is the last important stuff which you can learn from this

## Week-3

### Transfer Learning

Transfer learning is one of the most important techniques of deep learning and tensorflow lets us to do that in just small lines of code and We can download the open-source model that someone else already trained on the huge dataset and use that for our particular dataset like cats vs dogs,horse vs humans

doubt here........

We usually create our own cnn model which will be trained in particular image dataset and then cnn model will extract the features and we will be use that model for the classification but whereas in the transfer learning we will be having a already existing model where we can use that particular model which had already had trained features for huge dataset for example imagenet dataset with 1000 class variables and here we will be using that particular model to train in our model and use for prediction

### Lets make our hands dirty

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%201.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%201.png)

In order to do the transfer learning in the tesorflow we are using the tensorflow and keras it has layers api which helps us to make which layers we need to train and which layers to freeze

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%202.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%202.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%203.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%203.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%204.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%204.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%205.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%205.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%206.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%206.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%207.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%207.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%208.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%208.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%209.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%209.png)

Here this is the another type of overfitting condition where the validation accuracy is much less than the training accuracy so how to fix this 

To fix any overfitting condition we should use dropouts, regularisations,batchnormalisation and try changing the another model or even we can increase the training size of the images

We are going to use the dropouts layers because the model is overfitting even though we had included the image augumentaiton in it

So the idea behind the drop out is something like we will be having the same weights for both the classes and hence which inturn leads to the overfitting condition

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2010.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2010.png)

Here is how we achieve this in the tensorflow 

From this to 

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2011.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2011.png)

this

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2012.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2012.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2013.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2013.png)

# Week-4

## Multi-Class Classification

Till now we have been into the binary classification which is nothing but do we need to classify 2 classes but whereas now we are going to classify more than 2 classes

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2014.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2014.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2015.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2015.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2016.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2016.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2017.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2017.png)

![Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2018.png](Convolutional%20Neural%20Networks%20in%20TensorFlow%20d1f7ebc2756841098f6b06b7c9f34fa2/Untitled%2018.png)