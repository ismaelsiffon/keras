# Final Assignment: Build a Regression Model in Keras

Imports:


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

Let's download the data:


```python
concrete_data= pd.read_csv('https://cocl.us/concrete_data')
concrete_data.head()
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
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
      <th>Strength</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>




```python
concrete_data.shape
```




    (1030, 9)



Let's split the data into the predictors columns and the target column:


```python
df_columns = concrete_data.columns
predictors = concrete_data[df_columns[0:8]]
target = concrete_data[df_columns[8]]
```


```python
predictors.head()
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
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
target.head()
```




    0    79.99
    1    61.89
    2    40.27
    3    41.05
    4    44.30
    Name: Strength, dtype: float64



Let's normalize the data:


```python
n_cols = predictors_norm.shape[1]
```

Let's import Keras and the packages we need


```python
import keras
from keras.models import Sequential
from keras.layers import Dense
```

    Using TensorFlow backend.
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:521: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:522: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


# Part A: Build a baseline model

Let's define a function that calls the regression model:


```python
def regression_model_1 ():
    #create the model:
    model = Sequential()
    model.add(Dense(10,activation='relu',input_shape=(n_cols,)))
    model.add(Dense(1))
    
    #compile model:
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

model_1 = regression_model_1()
```

Let's fit and evaluate the model:


```python
n=50
MSE_1=[]
num_epochs = 50
for i in range(0,n):
    #Split the data (30% left for test data)
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=None, shuffle=True)

    #Fit the model with the train data:
    model_1.fit(X_train, y_train, epochs=num_epochs, verbose=0)

    #Predict using the test data:
    y_test_hat = model_1.predict(X_test)
    
    #Calculate and add the MSE to the MSE list:
    MSE_1.append(mean_squared_error(y_test, y_test_hat))

# Calculate the mean and the standard deviation of the MSE's:
MSE_mean = np.mean(MSE_1)
MSE_std= np.std(MSE_1)

print("A list of ", len(MSE_1), " mean square error values was created. The first 5 values are: ", MSE_1[0:5])
print("The MSE mean is ", MSE_mean)
print("The MSE standard deviation is ", MSE_std)
```

    A list of  50  mean square error values was created. The first 5 values are:  [2303926.579805271, 2344238.638592759, 2361534.706876994, 2338068.0522669023, 2365350.7032558755]
    The MSE mean is  2371416.187988514
    The MSE standard deviation is  64431.48876362147


# Part B: Normalize the data

Let's normalize the predictors:


```python
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
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
      <th>Cement</th>
      <th>Blast Furnace Slag</th>
      <th>Fly Ash</th>
      <th>Water</th>
      <th>Superplasticizer</th>
      <th>Coarse Aggregate</th>
      <th>Fine Aggregate</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>0.862735</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.476712</td>
      <td>-0.856472</td>
      <td>-0.846733</td>
      <td>-0.916319</td>
      <td>-0.620147</td>
      <td>1.055651</td>
      <td>-1.217079</td>
      <td>-0.279597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>3.551340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.491187</td>
      <td>0.795140</td>
      <td>-0.846733</td>
      <td>2.174405</td>
      <td>-1.038638</td>
      <td>-0.526262</td>
      <td>-2.239829</td>
      <td>5.055221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.790075</td>
      <td>0.678079</td>
      <td>-0.846733</td>
      <td>0.488555</td>
      <td>-1.038638</td>
      <td>0.070492</td>
      <td>0.647569</td>
      <td>4.976069</td>
    </tr>
  </tbody>
</table>
</div>



Let's fit and evaluate the same model, but this time using the normalized predictors:


```python
n=50
MSE_2=[]
num_epochs = 50
for i in range(0,n):
    #Split the data (30% left for test data)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=None, shuffle=True)

    #Fit the model with the train data:
    model_1.fit(X_train, y_train, epochs=num_epochs, verbose=0)

    #Predict using the test data:
    y_test_hat = model_1.predict(X_test)
    
    #Calculate and add the MSE to the MSE list:
    MSE_2.append(mean_squared_error(y_test, y_test_hat))

# Calculate the mean and the standard deviation of the MSE's:
MSE_2_mean = np.mean(MSE_2)
MSE_2_std= np.std(MSE_2)

print("A list of ", len(MSE_2), " mean square error values was created. The first 5 values are: ", MSE_2[0:5])
print("The MSE mean is ", MSE_2_mean)
print("The MSE standard deviation is ", MSE_2_std)
```

    A list of  50  mean square error values was created. The first 5 values are:  [65.15794383036972, 63.61739194147764, 64.88191165788174, 66.55539100247165, 66.77993470264741]
    The MSE mean is  65.36342628815278
    The MSE standard deviation is  4.899343345648107


### How does the mean of the mean squared errors compare to that from Step A?

Answer: We achieved a much smaller MSE compared to step A when using the normalized data. Having the predictors normalized before fitting the model resulted in a better performing model. 

# Part C: Increase the number of epochs:

Let's fit and evaluate the same model, but this time using 100 epochs:


```python
n=50
MSE_3=[]
num_epochs = 100
for i in range(0,n):
    #Split the data (30% left for test data)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=None, shuffle=True)

    #Fit the model with the train data:
    model_1.fit(X_train, y_train, epochs=num_epochs, verbose=0)

    #Predict using the test data:
    y_test_hat = model_1.predict(X_test)
    
    #Calculate and add the MSE to the MSE list:
    MSE_3.append(mean_squared_error(y_test, y_test_hat))

# Calculate the mean and the standard deviation of the MSE's:
MSE_3_mean = np.mean(MSE_3)
MSE_3_std= np.std(MSE_3)

print("A list of ", len(MSE_3), " mean square error values was created. The first 5 values are: ", MSE_3[0:5])
print("The MSE mean is ", MSE_3_mean)
print("The MSE standard deviation is ", MSE_3_std)
```

    A list of  50  mean square error values was created. The first 5 values are:  [51.080765585248, 51.131402256620326, 50.64708317874586, 43.60643558293141, 44.27955582562034]
    The MSE mean is  36.41592427699724
    The MSE standard deviation is  5.595864746545465


### How does the mean of the mean squared errors compare to that from Step B?

Answer: Inceasing the number of epochs resulted in a smaller MSE compared to step B. Although the process took much longer, the MSE went from 65 to 36, which is a considerable improvement for our estimator.

# Part D: Increase the number of hidden layers

Let's change our model to 3 hidden layers:


```python
def regression_model_2 ():
    #create the model:
    model2 = Sequential()
    model2.add(Dense(10,activation='relu',input_shape=(n_cols,)))
    model2.add(Dense(10,activation='relu'))
    model2.add(Dense(10,activation='relu'))
    model2.add(Dense(1))
    
    #compile model:
    model2.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

model_2 = regression_model_2()
```

Let's fit and evaluate the new model:


```python
n=50
MSE_4=[]
num_epochs = 50
for i in range(0,n):
    #Split the data (30% left for test data)
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=None, shuffle=True)

    #Fit the model with the train data:
    model_2.fit(X_train, y_train, epochs=num_epochs, verbose=0)

    #Predict using the test data:
    y_test_hat = model_2.predict(X_test)
    
    #Calculate and add the MSE to the MSE list:
    MSE_4.append(mean_squared_error(y_test, y_test_hat))

# Calculate the mean and the standard deviation of the MSE's:
MSE_4_mean = np.mean(MSE_4)
MSE_4_std= np.std(MSE_4)

print("A list of ", len(MSE_4), " mean square error values was created. The first 5 values are: ", MSE_4[0:5])
print("The MSE mean is ", MSE_4_mean)
print("The MSE standard deviation is ", MSE_4_std)
```

    A list of  50  mean square error values was created. The first 5 values are:  [32.423078581566216, 33.06681404470725, 35.25299212137717, 32.438555572138604, 33.07642328532886]
    The MSE mean is  31.714664245382934
    The MSE standard deviation is  2.5212655667321413


### How does the mean of the mean squared errors compare to that from Step B?

Answer: Adding hidden layers to the network resulted in a smaller MSE compared to step B. We observed a better performance compare to increasing the number of epochs. The process is faster and the performance better.
