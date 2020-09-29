# Sequences, Time Series and Prediction

In this particular course we are going to see the time series sequence model where the data changes over a period of time and it may be closing price of stocks or rainfall at particular day

### Week-1

***Time Series example***

The process of predicting the backward value like we have data from 1950-2000 and we need to predict the data at particular year suppose like 1930 and this is called as Imputed Data and otherwise is called is called Forecasts

Time series patterns are as follows 

This is an upwards faccing trend

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled.png)

The seasonalities it shows the active users in a website of software developers

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%201.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%201.png)

Then here comes something wierd combination of both trend and seasonalities having peaks and drops

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%202.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%202.png)

The below is typically called as white noise

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%203.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%203.png)

No seasonalities and no general vision and clearly all series are random

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%204.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%204.png)

Multiple Auto correlations

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%205.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%205.png)

Time series which we can encounter in our real life is something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%206.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%206.png)

And now as a machine learning part we need to forecase the learned pattern like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%207.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%207.png)

And yes real life time series are not always same they vary like this called as Non stationary Time series

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%208.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%208.png)

Here we are using that we will take the last few data in the big event so we can make a good prediction model which will help us to forecast better

***Intro to time series***

We are seeing the different time series and here gives the idea about the seasonalities auto correlations and noises and here is the link to the notebooks

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP_Week_1_Lesson_2.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_1_Lesson_2.ipynb)

### Train,validation and test sets

Naive Forecasting: -take the last value and assume that the next value will be the same one, and this is called naive forecasting  

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%209.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%209.png)

We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period. And this is called roll-forward partitioning

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2010.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2010.png)

Metrics for evaluating performance

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2011.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2011.png)

And if we want the mean of our errors' calculation to be of the same scale as the original errors, then we just get its square root, giving us a root means squared error or rmse. Another common metric and one of my favorites is the mean absolute error or mae, and it's also called the main absolute deviation or mad. And in this case, instead of squaring to get rid of negatives, it just uses their absolute value. This does not penalize large errors as much as the mse does. Depending on your task, you may prefer the mae or the mse. For example, if large errors are potentially dangerous and they cost you much more than smaller errors, then you may prefer the mse. But if your gain or your loss is just proportional to the size of the error, then the mae may be better.

Also, you can measure the mean absolute percentage error or mape, this is the mean ratio between the absolute error and the absolute value, this gives an idea of the size of the errors compared to the values.

```python
keras.metrics.mean_absolute_error(x_valid,naive_forecast).numpy()
```

**Moving average and differencing**

This particular method will convert the timeseries data into noiseless and thereby giving the average of actual data in particular period lets say 30 days

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2012.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2012.png)

Depending on the current time i.e. the period after which you want to forecast for the future, it can actually end up being worse than a naive forecast

**Differencing : -** One method to avoid this is to remove the trend and seasonality from the time series with a technique called differencing. So instead of studying the time series itself, we study the difference between the value at time T and the value at an earlier period.

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2013.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2013.png)

Depending on the time of your data, that period might be a year, a day, a month or whatever. Let's look at a year earlier. So for this data, at time T minus 365, we'll get this difference time series which has no trend and no seasonality. We can then use a moving average to forecast this time series which gives us these forecasts. But these are just forecasts for the difference time series, not the original time series. To get the final forecasts for the original time series, we just need to add back the value at time T minus 365, and we'll get these forecasts.

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2014.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2014.png)

You may have noticed that our moving average removed a lot of noise but our final forecasts are still pretty noisy. Where does that noise come from? Well, that's coming from the past values that we added back into our forecasts. So we can improve these forecasts by also removing the past noise using a moving average on that. If we do that, we get much smoother forecasts.

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2015.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2015.png)

**Trailing versus centered windows : -** Note that when we use the trailing window when computing the moving average of present values from t minus 32, t minus one. But when we use a centered window to compute the moving average of past values from one year ago, that's t minus one year minus five days, to t minus one year plus five days. Then moving averages using centered windows can be more accurate than using trailing windows. But we can't use centered windows to smooth present values since we don't know future values. However, to smooth past values we can afford to use centered windows.

**Forecasting**

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 1 - Lesson 3 - Notebook.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%201%20-%20Lesson%203%20-%20Notebook.ipynb)

## Week - 2

Preparing features and labels: -  

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.windows(5, shift=1,drop_remainder=True)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(),end=" ")
    print()
```

The output is something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2016.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2016.png)

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.windows(5, shift=1,drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1] , window[-1:]) )
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print(x.numpy(),y.numpy())
```

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 2 Lesson 1.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%201.ipynb)

### Feading windowed dataset into neural network

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2017.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2017.png)

Single Layer neural network

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2018.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2018.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2019.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2019.png)

The above mentioned neural network is simply a linear regression and for compiling we are using mse as a loss and sgd optimiser and epochs to 100

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2020.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2020.png)

The output of the weights and the bias of the linear regression model is something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2021.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2021.png)

The top array is the weights and next is the bias

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2022.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2022.png)

Plotting the forecast is some thing like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2023.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2023.png)

The mean absolute error is calculated as follows

```python
tf.keras.metrics.mean_absolute_error(x_valid,results).numpy()

# 4.9525777
```

***Lets go to the deep Neural Network***

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2024.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2024.png)

The error is 4.9833784

Let us play with the learning rate scheduler to the optimizers like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2025.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2025.png)

Plotting the loss per epoch and epoch value like this 

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2026.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2026.png)

The graph is like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2027.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2027.png)

From this we understood that this is the better learning rate something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2028.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2028.png)

The loss is something like this 4.4847784

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2029.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2029.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2030.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2030.png)

The result of the graph is something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2031.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2031.png)

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 2 Lesson 3.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%203.ipynb)

### Week -3

***Conceptual Overview***

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2032.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2032.png)

The rnn layer is some thing like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2033.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2033.png)

***Shape of the inputs to the RNN***

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2034.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2034.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2035.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2035.png)

Suppose if we don't need the output of each states and need only the final output something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2036.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2036.png)

Outputting a sequence

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2037.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2037.png)

If return_sequence = True will be like this and this is called as a sequence to sequence rnn model where the input will be a sequence and the output will also be sequence

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2038.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2038.png)

***Lambda Layers:***

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2039.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2039.png)

***Adjusting the learning rate dynamically***

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2040.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2040.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2041.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2041.png)

The notebook is available here

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 3 Lesson 2 - RNN.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%202%20-%20RNN.ipynb)

***LSTM:***

Normal LSTM with CellState

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2042.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2042.png)

BiDirectional LSTM

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2043.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2043.png)

The complete tutorials on the LSTM is here

[***https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay***](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)

Coding 

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2044.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2044.png)

The output is something like this

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2045.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2045.png)

Let us add another Bidirectional layer 

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2046.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2046.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2047.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2047.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2048.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2048.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2049.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2049.png)

This is bad.........

And please find the notebooks here

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 3 Lesson 4 - LSTM.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%204%20-%20LSTM.ipynb)

### Week -4

Convolution

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2050.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2050.png)

Bi-directional LSTMs

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2051.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2051.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2052.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2052.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2053.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2053.png)

Make it Bidirectional

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2054.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2054.png)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2055.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2055.png)

But it is overfitting and this is the graph we are plotting on validation set

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2056.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2056.png)

See how the batch size works here

[***https://www.youtube.com/watch?v=4qJaSmvhxi8***](https://www.youtube.com/watch?v=4qJaSmvhxi8)

The note book is here

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 4 Lesson 1.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%201.ipynb)

The notebook for the sunspot solution with DNN is here

[***https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 4 - S%2BP/S%2BP Week 4 Lesson 5.ipynb***](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%205.ipynb)

![Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2057.png](Sequences,%20Time%20Series%20and%20Prediction%205cd54ac85f344fd49f3b0ade46ad237d/Untitled%2057.png)