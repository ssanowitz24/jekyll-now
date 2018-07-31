
# Fun with Neural Nets
Predicting handwritten numerical digits with a data set from kaggle and utilizing neural network models


---
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.


In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

# Kaggle Procedure

Below is a procedure for building a neural network to recognize handwritten digits.  The data is from Kaggle, and you will submit your results to Kaggle to test how well you did!

1. Load the training data (`train.csv`) from Kaggle
2. Setup X and y (feature matrix and target vector)
3. Split X and y into train and test subsets.
4. Preprocess your data

   - When dealing with image data, you need to normalize your `X` by dividing each value by the max value of a pixel (255).
   - Since this is a multiclass classification problem, keras needs `y` to be a one-hot encoded matrix
   
5. Create your network.

   - Remember that for multi-class classification you need a softamx activation function on the output layer.
   - You may want to consider using regularization or dropout to improve performance.
   
6. Trian your network.
7. If you are unhappy with your model performance, try to tighten up your model by adding hidden layers, adding hidden layer units, chaning the activation functions on the hidden layers, etc.
8. Load in Kaggle's `test.csv`
9. Create your predictions (these should be numbers in the range 0-9).
10. Save your predictions and submit them to Kaggle.

# Code


```python
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
```


```python
# read in data
training_data = pd.read_csv('./train.csv')
```


```python
training_data.head()
#targets
sorted(training_data.label.unique())

```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
#target and features
y = training_data['label']
feats= [ col for col in training_data.columns if col != 'label' ]
X = training_data[feats]
```


```python
X.shape
y.shape

```




    (42000,)




```python
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=55)
```


```python
X_train.shape
y_train.shape
X_test.shape
y_test.shape
```




    (10500,)




```python
#normalizing hexadecimal
X_train /= 255

```


```python
X_test /= 255
```


```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
```

    /Users/scottsanowitz/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


    Couldn't import dot_parser, loading of dot files will not be possible.



```python
#target matrix for nueral net
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
```


```python
y_train.shape
y_test.shape
```




    (10500, 10)




```python
#model
model = Sequential()
```


```python
# layers,
n_input = X_train.shape[1]
n_hidden = n_input
#hidden layer
model.add(Dense(n_input, activation='relu', input_dim=n_input))
# out put layer with softmax activation
model.add(Dense(10, activation='softmax'))
```


```python
# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
```


```python
#fit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs =50, batch_size=2000)
```

    Train on 31500 samples, validate on 10500 samples
    Epoch 1/50
    31500/31500 [==============================] - 3s 87us/step - loss: 1.0081 - acc: 0.7369 - val_loss: 0.4217 - val_acc: 0.8803
    Epoch 2/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.3520 - acc: 0.8965 - val_loss: 0.3058 - val_acc: 0.9156
    Epoch 3/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.2721 - acc: 0.9210 - val_loss: 0.2596 - val_acc: 0.9288
    Epoch 4/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.2310 - acc: 0.9347 - val_loss: 0.2304 - val_acc: 0.9360
    Epoch 5/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.2010 - acc: 0.9439 - val_loss: 0.2059 - val_acc: 0.9424
    Epoch 6/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.1777 - acc: 0.9504 - val_loss: 0.1907 - val_acc: 0.9463
    Epoch 7/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.1577 - acc: 0.9571 - val_loss: 0.1734 - val_acc: 0.9518
    Epoch 8/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.1413 - acc: 0.9619 - val_loss: 0.1626 - val_acc: 0.9533
    Epoch 9/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.1276 - acc: 0.9657 - val_loss: 0.1552 - val_acc: 0.9553
    Epoch 10/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.1161 - acc: 0.9688 - val_loss: 0.1441 - val_acc: 0.9596
    Epoch 11/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.1029 - acc: 0.9726 - val_loss: 0.1358 - val_acc: 0.9622
    Epoch 12/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.0938 - acc: 0.9754 - val_loss: 0.1289 - val_acc: 0.9638
    Epoch 13/50
    31500/31500 [==============================] - 3s 87us/step - loss: 0.0859 - acc: 0.9777 - val_loss: 0.1246 - val_acc: 0.9643
    Epoch 14/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.0776 - acc: 0.9802 - val_loss: 0.1210 - val_acc: 0.9661
    Epoch 15/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.0715 - acc: 0.9821 - val_loss: 0.1164 - val_acc: 0.9665
    Epoch 16/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.0658 - acc: 0.9831 - val_loss: 0.1141 - val_acc: 0.9677
    Epoch 17/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.0605 - acc: 0.9856 - val_loss: 0.1098 - val_acc: 0.9700
    Epoch 18/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.0559 - acc: 0.9864 - val_loss: 0.1087 - val_acc: 0.9690
    Epoch 19/50
    31500/31500 [==============================] - 3s 81us/step - loss: 0.0508 - acc: 0.9883 - val_loss: 0.1055 - val_acc: 0.9703
    Epoch 20/50
    31500/31500 [==============================] - 3s 82us/step - loss: 0.0462 - acc: 0.9894 - val_loss: 0.1041 - val_acc: 0.9704
    Epoch 21/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0434 - acc: 0.9904 - val_loss: 0.1016 - val_acc: 0.9712
    Epoch 22/50
    31500/31500 [==============================] - 3s 92us/step - loss: 0.0390 - acc: 0.9918 - val_loss: 0.0998 - val_acc: 0.9729
    Epoch 23/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0359 - acc: 0.9927 - val_loss: 0.0990 - val_acc: 0.9719
    Epoch 24/50
    31500/31500 [==============================] - 3s 96us/step - loss: 0.0336 - acc: 0.9936 - val_loss: 0.0985 - val_acc: 0.9726
    Epoch 25/50
    31500/31500 [==============================] - 3s 85us/step - loss: 0.0310 - acc: 0.9942 - val_loss: 0.0973 - val_acc: 0.9731
    Epoch 26/50
    31500/31500 [==============================] - 3s 87us/step - loss: 0.0285 - acc: 0.9951 - val_loss: 0.0960 - val_acc: 0.9729
    Epoch 27/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0269 - acc: 0.9955 - val_loss: 0.0952 - val_acc: 0.9738
    Epoch 28/50
    31500/31500 [==============================] - 3s 87us/step - loss: 0.0253 - acc: 0.9962 - val_loss: 0.0942 - val_acc: 0.9734
    Epoch 29/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0230 - acc: 0.9965 - val_loss: 0.0933 - val_acc: 0.9732
    Epoch 30/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0219 - acc: 0.9966 - val_loss: 0.0951 - val_acc: 0.9734
    Epoch 31/50
    31500/31500 [==============================] - 3s 83us/step - loss: 0.0205 - acc: 0.9972 - val_loss: 0.0939 - val_acc: 0.9740
    Epoch 32/50
    31500/31500 [==============================] - 3s 85us/step - loss: 0.0186 - acc: 0.9977 - val_loss: 0.0929 - val_acc: 0.9744
    Epoch 33/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0176 - acc: 0.9978 - val_loss: 0.0923 - val_acc: 0.9743
    Epoch 34/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0162 - acc: 0.9983 - val_loss: 0.0916 - val_acc: 0.9738
    Epoch 35/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0156 - acc: 0.9983 - val_loss: 0.0917 - val_acc: 0.9745
    Epoch 36/50
    31500/31500 [==============================] - 3s 86us/step - loss: 0.0143 - acc: 0.9987 - val_loss: 0.0920 - val_acc: 0.9739
    Epoch 37/50
    31500/31500 [==============================] - 3s 88us/step - loss: 0.0134 - acc: 0.9987 - val_loss: 0.0917 - val_acc: 0.9739
    Epoch 38/50
    31500/31500 [==============================] - 3s 93us/step - loss: 0.0126 - acc: 0.9991 - val_loss: 0.0920 - val_acc: 0.9747
    Epoch 39/50
    31500/31500 [==============================] - 3s 91us/step - loss: 0.0120 - acc: 0.9993 - val_loss: 0.0910 - val_acc: 0.9746
    Epoch 40/50
    31500/31500 [==============================] - 3s 92us/step - loss: 0.0111 - acc: 0.9992 - val_loss: 0.0914 - val_acc: 0.9749
    Epoch 41/50
    31500/31500 [==============================] - 3s 93us/step - loss: 0.0105 - acc: 0.9994 - val_loss: 0.0913 - val_acc: 0.9745
    Epoch 42/50
    31500/31500 [==============================] - 3s 94us/step - loss: 0.0100 - acc: 0.9994 - val_loss: 0.0906 - val_acc: 0.9748
    Epoch 43/50
    31500/31500 [==============================] - 3s 93us/step - loss: 0.0097 - acc: 0.9995 - val_loss: 0.0913 - val_acc: 0.9749
    Epoch 44/50
    31500/31500 [==============================] - 3s 92us/step - loss: 0.0091 - acc: 0.9997 - val_loss: 0.0918 - val_acc: 0.9749
    Epoch 45/50
    31500/31500 [==============================] - 3s 92us/step - loss: 0.0085 - acc: 0.9997 - val_loss: 0.0915 - val_acc: 0.9755
    Epoch 46/50
    31500/31500 [==============================] - 3s 92us/step - loss: 0.0079 - acc: 0.9997 - val_loss: 0.0918 - val_acc: 0.9752
    Epoch 47/50
    31500/31500 [==============================] - 3s 100us/step - loss: 0.0077 - acc: 0.9997 - val_loss: 0.0925 - val_acc: 0.9754
    Epoch 48/50
    31500/31500 [==============================] - 3s 90us/step - loss: 0.0073 - acc: 0.9998 - val_loss: 0.0925 - val_acc: 0.9749
    Epoch 49/50
    31500/31500 [==============================] - 3s 91us/step - loss: 0.0069 - acc: 0.9998 - val_loss: 0.0912 - val_acc: 0.9757
    Epoch 50/50
    31500/31500 [==============================] - 3s 95us/step - loss: 0.0065 - acc: 0.9999 - val_loss: 0.0912 - val_acc: 0.9759



```python
#predict
y_hat= model.predict(X_test)
```


```python
df =pd.DataFrame(y_hat)
df.index = df.index + 1
```


```python
df['max'] = df.idxmax(axis=1)
```


```python
#ready for kaggle
df[['max']].index.name = 'ImageId'
df[['max']].rename({'max': 'Label'},axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
    </tr>
    <tr>
      <th>ImageId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>8</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
    </tr>
    <tr>
      <th>30</th>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>10471</th>
      <td>7</td>
    </tr>
    <tr>
      <th>10472</th>
      <td>6</td>
    </tr>
    <tr>
      <th>10473</th>
      <td>1</td>
    </tr>
    <tr>
      <th>10474</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10475</th>
      <td>8</td>
    </tr>
    <tr>
      <th>10476</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10477</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10478</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10479</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10480</th>
      <td>7</td>
    </tr>
    <tr>
      <th>10481</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10482</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10483</th>
      <td>8</td>
    </tr>
    <tr>
      <th>10484</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10485</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10486</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10487</th>
      <td>7</td>
    </tr>
    <tr>
      <th>10488</th>
      <td>2</td>
    </tr>
    <tr>
      <th>10489</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10490</th>
      <td>1</td>
    </tr>
    <tr>
      <th>10491</th>
      <td>6</td>
    </tr>
    <tr>
      <th>10492</th>
      <td>4</td>
    </tr>
    <tr>
      <th>10493</th>
      <td>6</td>
    </tr>
    <tr>
      <th>10494</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10495</th>
      <td>4</td>
    </tr>
    <tr>
      <th>10496</th>
      <td>6</td>
    </tr>
    <tr>
      <th>10497</th>
      <td>2</td>
    </tr>
    <tr>
      <th>10498</th>
      <td>2</td>
    </tr>
    <tr>
      <th>10499</th>
      <td>9</td>
    </tr>
    <tr>
      <th>10500</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10500 rows × 1 columns</p>
</div>




```python
test = pd.read_csv('./test.csv')
```


```python
test/=255
```


```python
test.shape
```




    (28000, 784)




```python
y_hat = model.predict(test)
```


```python
tested =pd.DataFrame(y_hat)
tested.index = tested.index +1
```


```python
tested['max'] = tested.idxmax(axis=1)
```


```python
tested['max']
```




    1        2
    2        0
    3        9
    4        9
    5        3
    6        7
    7        0
    8        3
    9        0
    10       3
    11       5
    12       7
    13       4
    14       0
    15       4
    16       3
    17       3
    18       1
    19       9
    20       0
    21       9
    22       1
    23       1
    24       5
    25       7
    26       4
    27       2
    28       7
    29       4
    30       7
            ..
    27971    5
    27972    0
    27973    4
    27974    8
    27975    0
    27976    3
    27977    6
    27978    0
    27979    1
    27980    9
    27981    3
    27982    1
    27983    1
    27984    0
    27985    4
    27986    5
    27987    2
    27988    2
    27989    9
    27990    6
    27991    7
    27992    6
    27993    1
    27994    9
    27995    7
    27996    9
    27997    7
    27998    3
    27999    9
    28000    2
    Name: max, Length: 28000, dtype: int64




```python
#submit to kaggle
tested[['max']].index.name = 'ImageId'
tested['ImageId'] = tested.index
tested
submit = tested[['ImageId','max']].rename({'max': 'Label'},axis=1)
submit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
    <tr>
      <th>ImageId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>9</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>7</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27971</th>
      <td>27971</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27972</th>
      <td>27972</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27973</th>
      <td>27973</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27974</th>
      <td>27974</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27975</th>
      <td>27975</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27976</th>
      <td>27976</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27977</th>
      <td>27977</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27978</th>
      <td>27978</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27979</th>
      <td>27979</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27980</th>
      <td>27980</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27981</th>
      <td>27981</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27982</th>
      <td>27982</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27983</th>
      <td>27983</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27984</th>
      <td>27984</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27985</th>
      <td>27985</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27986</th>
      <td>27986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27987</th>
      <td>27987</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27988</th>
      <td>27988</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27989</th>
      <td>27989</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27990</th>
      <td>27990</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27991</th>
      <td>27991</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27992</th>
      <td>27992</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27993</th>
      <td>27993</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27994</th>
      <td>27994</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27995</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28000</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```python
submit.to_csv('./submit.csv',index=False)
```

# CNN
Convolutional Neural Net


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=124)
```


```python
X_train = X_train.values.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

```


```python
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
```


```python
X_train /= 255
X_test /= 255
```


```python
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
```


```python
model = Sequential()
```


```python
#convolution layers
model.add(Convolution2D(filters =16,
                       kernel_size = 3,
                       activation ='relu',
                       input_shape = (28,28,1)))
#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
#convolution layer
model.add(Convolution2D(filters =20,
                       kernel_size = 3,
                       activation ='relu'
                       ))
#pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
#flatten layer
model.add(Flatten())
#dense layers
model.add(Dense(150, activation='relu'))
#output layer
model.add(Dense(10, activation='softmax'))
```


```python
#compile
model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics=['acc'] )
early_stop = EarlyStopping(monitor='val_loss', min_delta=0)
```


```python
#fit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs =50, batch_size=2000, callbacks=[early_stop])
```

    Train on 31500 samples, validate on 10500 samples
    Epoch 1/50
    31500/31500 [==============================] - 11s 338us/step - loss: 1.9931 - acc: 0.4954 - val_loss: 1.4562 - val_acc: 0.7004
    Epoch 2/50
    31500/31500 [==============================] - 10s 315us/step - loss: 0.9485 - acc: 0.7736 - val_loss: 0.5732 - val_acc: 0.8293
    Epoch 3/50
    31500/31500 [==============================] - 10s 315us/step - loss: 0.4474 - acc: 0.8661 - val_loss: 0.3702 - val_acc: 0.8866
    Epoch 4/50
    31500/31500 [==============================] - 10s 315us/step - loss: 0.3208 - acc: 0.9034 - val_loss: 0.2830 - val_acc: 0.9159
    Epoch 5/50
    31500/31500 [==============================] - 10s 322us/step - loss: 0.2548 - acc: 0.9236 - val_loss: 0.2337 - val_acc: 0.9301
    Epoch 6/50
    31500/31500 [==============================] - 10s 319us/step - loss: 0.2136 - acc: 0.9372 - val_loss: 0.1992 - val_acc: 0.9392
    Epoch 7/50
    31500/31500 [==============================] - 10s 320us/step - loss: 0.1804 - acc: 0.9471 - val_loss: 0.1701 - val_acc: 0.9505
    Epoch 8/50
    31500/31500 [==============================] - 10s 320us/step - loss: 0.1536 - acc: 0.9552 - val_loss: 0.1472 - val_acc: 0.9570
    Epoch 9/50
    31500/31500 [==============================] - 10s 321us/step - loss: 0.1328 - acc: 0.9612 - val_loss: 0.1306 - val_acc: 0.9612
    Epoch 10/50
    31500/31500 [==============================] - 10s 321us/step - loss: 0.1166 - acc: 0.9653 - val_loss: 0.1163 - val_acc: 0.9661
    Epoch 11/50
    31500/31500 [==============================] - 11s 353us/step - loss: 0.1045 - acc: 0.9693 - val_loss: 0.1073 - val_acc: 0.9679
    Epoch 12/50
    31500/31500 [==============================] - 10s 321us/step - loss: 0.0955 - acc: 0.9718 - val_loss: 0.1015 - val_acc: 0.9700
    Epoch 13/50
    31500/31500 [==============================] - 10s 320us/step - loss: 0.0894 - acc: 0.9730 - val_loss: 0.0955 - val_acc: 0.9716
    Epoch 14/50
    31500/31500 [==============================] - 10s 321us/step - loss: 0.0814 - acc: 0.9753 - val_loss: 0.0871 - val_acc: 0.9730
    Epoch 15/50
    31500/31500 [==============================] - 10s 322us/step - loss: 0.0751 - acc: 0.9776 - val_loss: 0.0808 - val_acc: 0.9743
    Epoch 16/50
    31500/31500 [==============================] - 12s 366us/step - loss: 0.0700 - acc: 0.9791 - val_loss: 0.0828 - val_acc: 0.9746





    <keras.callbacks.History at 0x1a44687b70>




```python
test = pd.read_csv('./test.csv')
```


```python
test = test.values.reshape(test.shape[0], 28, 28, 1)
```


```python
test = test.astype('float32')
```


```python
test /= 255
```


```python
y_hat = model.predict_classes(test)
```


```python
y_hat
```




    array([2, 0, 9, ..., 3, 9, 2])




```python
tested =pd.DataFrame(y_hat)
tested.index = tested.index +1
```


```python
tested.rename({0:'Label'}, axis =1, inplace=True)
tested['ImageId'] = tested.index

```


```python
submit2 = tested[['ImageId','Label']]

submit2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>9</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>7</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>7</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27971</th>
      <td>27971</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27972</th>
      <td>27972</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27973</th>
      <td>27973</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27974</th>
      <td>27974</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27975</th>
      <td>27975</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27976</th>
      <td>27976</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27977</th>
      <td>27977</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27978</th>
      <td>27978</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27979</th>
      <td>27979</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27980</th>
      <td>27980</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27981</th>
      <td>27981</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27982</th>
      <td>27982</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27983</th>
      <td>27983</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27984</th>
      <td>27984</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27985</th>
      <td>27985</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27986</th>
      <td>27986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27987</th>
      <td>27987</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27988</th>
      <td>27988</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27989</th>
      <td>27989</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27990</th>
      <td>27990</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27991</th>
      <td>27991</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27992</th>
      <td>27992</td>
      <td>6</td>
    </tr>
    <tr>
      <th>27993</th>
      <td>27993</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27994</th>
      <td>27994</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27995</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>28000</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>




```python
#submit to kaggle
submit2.to_csv('./submit2.csv', index=False)
```
# Conclusion
Learning about fully forward neural networks and convolutional neural networks this kaggle competition was great practice. I could see the power of the convolutional neural network on image data and it produces an respectable kaggle score. Out of all the models I have learned the Neural Networks is one of my favorites to build and test. 