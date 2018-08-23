# Deep Learning - Notebooks

## Getting Started

1. Download Miniconda [here](https://conda.io/miniconda.html)

2. Install Miniconda,
    - In Linux, run the downloaded `.sh` file using the following command:
    ```
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
    - In Windows, double-click the downloaded `.exe` file

3. Create a new environment in Miniconda using the provided `deep-learning.yml` file using the following command:
    ```
    conda env create -f deep-learning.yml
    ```
    This will create a new environment named `deep-learning`
    
4. Activate the created environment,
    - In Linux, use the following command:
    ```
    source activate deep-learning
    ```
    - In Windows, use the following command:
    ```
    activate deep-learning
    ```

5. Start a Jupyter notebook server by using the following command:
    ```
    jupyter notebook
    ```


## deep-learning Notebook

Summary of the basics of deep learning, including deep neural network. The notebook explains different kinds of activation functions and when to use them, feed forward and back propagation, weight initialization, dropout, and various optimizers for neural network.

**NOTE**: The notebook contains mathematical notation in latex that fails to show up if viewed outside `jupyter notebook` from Terminal.


## keras-mnist Notebook

<p align="center">
  <img width="480" height="360" src="https://camo.githubusercontent.com/d440ac2eee1cb3ea33340a2c5f6f15a0878e9275/687474703a2f2f692e7974696d672e636f6d2f76692f3051493378675875422d512f687164656661756c742e6a7067">
</p>

Deep learning experiment with Keras using the MNIST dataset from Keras. The MNIST dataset is a collection of handwritten digits (0-9) and the goal of the experiment is to correctly classify the handwriting based on the number it represents.

### Importing Dataset

The MNIST dataset is provided by Keras and can be imported by using the command below.

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Preprocessing

The data is first reshaped from the shape (28,28) to (764,) and normalized such that the value is in range [0..1].

```python
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
```

``` python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```

The categorical label is also one-hot encoded to further enhance the training process.
```python
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)
```

### Training

The training is done by calling the `fit` function on our model. The loss is calculated using cross-entropy with the Adam optimizer.

```python
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_limit,
                    verbose=1,
                    validation_data=(x_test, y_test))
```


### Testing

Testing is done by calling the `evaluate` function on our model. This will return the loss and the accuracy of the model during testing.

```python
score = model.evaluate(x_test, y_test,
                       verbose=1)
print('Test Loss = {}, Test Accuracy = {}'.format(score[0], score[1]))
```

### Analysis

Before analyzing the missed test cases, first we need to determine which test cases got classified incorrectly.

```python
y_test_prediction = model.predict(x_test, verbose=1)

# Round to turn predictions to 0 and 1
y_test_prediction = np.round(y_test_prediction)

failed_indices = []
false_count = 0
for i in range (len(y_test)):
    for j in range (n_class):
        if y_test_prediction[i][j] != y_test[i][j]:
            failed_indices.append(i)
            false_count += 1
            break
            
print("Total of {} failed test case".format(false_count))
```

The test cases are then randomly picked one-by-one, drawn, and analyzed to determine if a test case is indeed hard even for humans or it is easy for humans but the machine still guesses it incorrectly.


## Miscellaneous

To deactivate the miniconda environment,
- In Linux, use the following command:
    ```
    source deactivate
    ```
- In Windows, use the following command:
    ```
    deactivate
    ```