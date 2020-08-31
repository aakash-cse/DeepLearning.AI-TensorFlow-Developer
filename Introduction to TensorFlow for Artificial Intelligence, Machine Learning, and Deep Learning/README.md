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

## Assignment in this link
