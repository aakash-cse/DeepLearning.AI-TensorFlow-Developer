# Natural Language Processing in TensorFlow

This particular course deals with the text data where it differs by word length and sentence length. here it is not like cat vs dogs where images will have certain rgb pixel values here we are going to see how the data is converted in sensible number which can fed into the neural networks so that we can get good results

## Word Based encodings

We can take the character encoding in the each character in a set. For example, the ascii values.

suppose if we use character encoding for the words 'LISTEN' and 'SILENT" the word length as well as same no of the letters are repeated but the semantic is exactly different so the neural network failed to get the semantic of the word using character encoding

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled.png)

here the each and every word is given a value like label encoder and the dog has got the label 004 and now in the same word we have used cat which got 005 The similiarities of the two sentences are like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%201.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%201.png)

Any how the tensorflow, keras have given the API for manipulating these sentence to the labels like this.... here we go......

## Using APIs

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
	'I love my dog',
	'I love my cat'
]

tokenizer = Tokenizer=(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

here the point to be noticed is that the output word_index will something looks like this where the capitalised "I" will be lowercased

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%202.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%202.png)

Another case is that we have used the exclamation mark after dog and lets see what happens

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%203.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%203.png)

The output is something like this where the exclamation mark is completely removed

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%204.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%204.png)

## NoteBook for the above code

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
	'I love my dog',
	'I love my cat'
]

tokenizer = Tokenizer=(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

## Text to Sequence

The next step will be to turn your sentences into lists of values based on these tokens. Now we cannot know whether all the sentences will have same length or list which have same length. In images we have input size and we will resize the training images to the size of the images and train the neural network.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
	'I love my dog',
	'I love my cat',
	'You love my dog!',
	'Do you think my dog is amazing?'
]

tokenizer = Tokenizer=(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
```

The output is something like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%205.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%205.png)

But see this we have an issue

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%206.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%206.png)

The below code will make the unseen word index to have unique value.

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%207.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%207.png)

The output is something like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%208.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%208.png)

## Padding

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
	'I love my dog',
	'I love my cat',
	'You love my dog!',
	'Do you think my dog is amazing?'
]

tokenizer = Tokenizer=(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

padded = pad_sequences(sequences)
print(padded)
```

The output is something like this 

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%209.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%209.png)

We can change the pre zeros to post zeros using parameter padding='post' and change the length of the words like maxlen=5.Like with the padding the default is pre, which means that you will lose from the beginning of the sentence. If you want to override this so that you lose from the end instead, you can do so with truncating='post'

# Week -2

## Word Embedding

Word embedding is something like making the positive and negative associated words to particular thing like fun and associated words like funny have positive rating and unfunny and unfun will be negative and fundamental is something like neutral words....

In this topic we are going to discuss the imdb review dataset where we are using to find the word embedding and find whether the sentences is positive review or negative review or not

```python
import tensorflow as tf
print(tf.__version__)
'''
while using colab we should use the following sentences
!pip install -q tensorflow-datasets
'''

#importing the datasets
import tensorflow_datasets as tfds
imdb,info = tfds.load("imdb_reviews",with_info=True,as_supervised=True)

import numpy as np
train_data,test_data = imdb['train'],imdb['test']

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s,l in train_data:
	training_sentences.append(str(s.numpy()))
	training_labels.append(l.numpy())

for s,l in test_data:
	testing_data.append(str(s.numpy()))
	testing_labels.append(l.numpy())
```

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2010.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2010.png)

```python
model = tf.keras.Sequential([
						tf.keras.layers.Embedding(vocab_size,embedding_dim,
																				input_length=max_length),
						tf.keras.layers.Flatten(),
						tf.keras.layers.Dense(6,activation='relu'),
						tf.keras.layers.Dense(1,activation='sigmoid')
])
```

The embedding is used in the text sentimental analysis.

***How can we use vectors?***

You have words in a sentence and often words that have similar meanings are close to each other. So in a movie review, it might say that the movie was dull and boring, or it might say that it was fun and exciting. So what if you could pick a vector in a higher-dimensional space say 16 dimensions, and words that are found together are given similar vectors

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2011.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2011.png)

We can use **GlobalAveragePooling1D()** in place of **Flatten()**

We can compile the model like below

```python
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded,training_labels_final,epochs=num_epochs,
          validation_data=(testing_padded,testing_labels_final))
```

```python
#To plot the reversed word index of the word_index it is like this
reverse_word_index = dict([(value,key) 
                          for (key,value) in word_index.items()])
```

It's time to write the vectors and their metadata auto files. The TensorFlow Projector reads this file type and uses it to plot the vectors in 3D space so we can visualize them. To the vectors file, we simply write out the value of each of the items in the array of embeddings, i.e, the co-efficient of each dimension on the vector for this word. To the metadata array, we just write out the words

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2012.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2012.png)

while using colab use the following code to download the tsv files

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2013.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2013.png)

The notebook is here 

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 3 - NLP/Course 3 - Week 2 - Lesson 1.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb)

### Remeber the sarcasm dataset?

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2014.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2014.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2015.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2015.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2016.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2016.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2017.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2017.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2018.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2018.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2019.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2019.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2020.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2020.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2021.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2021.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2022.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2022.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2023.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2023.png)

This is good but the validation loss is increasing and so we can try for another loss function.

This can be done by twiking the hyperparameters inorder to decrease the loss

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2024.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2024.png)

This is the result of the code

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2025.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2025.png)

But now the accuracy is that much trying to change the size of the embedding

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2026.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2026.png)

This is the result of the following twiked hyperparameter

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2027.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2027.png)

The complete code of this is given below

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 3 - NLP/Course 3 - Week 2 - Lesson 2.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb)

### Sub-Tokens

please find the below [link](https://github.com/tensorflow/datasets/tree/master/docs/catalog)

The url shown for the imdb_reviews.md is [here](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2028.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2028.png)

```python
train_data,test_data = imdb['train'],imdb['test']

tokenizer = info.features['text'].encoder

#tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
```

Please find the url [here](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder)

```python
print(tokenizer.subwords)
```

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2029.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2029.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2030.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2030.png)

Here if we use Flatten it will crash 

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2031.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2031.png)

Plotting the graph is something like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2032.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2032.png)

The complete notebook is found [here](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb)

# Week-3

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2033.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2033.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2034.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2034.png)

This is the basic idea that how recurrent neural network works

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2035.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2035.png)

This is how the RNN actually looks like with a state being passed to next state

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2036.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2036.png)

***LSTM***

The updated RNN with Cell state is called as a LSTM (long-short term memory)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2037.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2037.png)

The LSTM can be bidirectional which looks like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2038.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2038.png)

Implementing the code

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2039.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2039.png)

We can stack more bidirectional lstm by passing return_sequence=True which make the output to be the input of the second bidirectional lstm layer

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2040.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2040.png)

Here is the code to the LSTM [single](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201a.ipynb) and [multilayer](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201b.ipynb)

***Comparison between the different layers of the LSTM like this***

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2041.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2041.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2042.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2042.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2043.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2043.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2044.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2044.png)

Let us code something better

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2045.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2045.png)

But we can experiment with the layers that bridge the embedding and the dense by removing the flatten and puling from here, and replacing them with an LSTM like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2046.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2046.png)

***The output is something like this***

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2047.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2047.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2048.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2048.png)

***Lets use convolutional network***

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2049.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2049.png)

The graph looks like this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2050.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2050.png)

The notebook is [here](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb)

The complete analysis of various lstm convolutional and gru are [here](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202d.ipynb#scrollTo=nHGYuU4jPYaj) 

Please find the below notebooks

[https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 3 - NLP/Course 3 - Week 3 - Lesson 2.ipynb#scrollTo=g9DC6dmLF8DC](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202.ipynb#scrollTo=g9DC6dmLF8DC)

[https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 3 - NLP/Course 3 - Week 3 - Lesson 2c.ipynb#scrollTo=g9DC6dmLF8DC](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%202c.ipynb#scrollTo=g9DC6dmLF8DC)

# Week -4

This is awesome week actually we are going to predict if we give input as "twinkle twinkle little " the output will be "star"

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2051.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2051.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2052.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2052.png)

***Training the data***

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2053.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2053.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2054.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2054.png)

```python
max_sequence_len = max(len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                    maxlen=max_sequence_len , padding = 'pre'))
```

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2055.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2055.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2056.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2056.png)

```python
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)
```

Please find the code [here](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%201%20-%20Notebook.ipynb)

***Finding what the next word should be*** 

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2057.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2057.png)

The accuracy is better

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2058.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2058.png)

Here is the output prediction

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2059.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2059.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2060.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2060.png)

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2061.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2061.png)

Here is the prediction but still there is some repetition in this

![Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2062.png](Natural%20Language%20Processing%20in%20TensorFlow%203b1c094716d5437b877f5536eef62b2a/Untitled%2062.png)

Please find the link to Laurences generated poetry [here.](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt)

Let's take a look at the second [notebook](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb) for this week.

## Link to generating text using a character-based RNN

[link](https://www.tensorflow.org/tutorials/sequences/text_generation)