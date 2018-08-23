# Generative Adversarial Network - Notebooks

## Getting Started

1. Download Miniconda [here](https://conda.io/miniconda.html)

2. Install Miniconda,
    - In Linux, run the downloaded `.sh` file using the following command:
    ```
    ./Miniconda3-latest-Linux-x86_64.sh
    ```
    - In Windows, double-click the downloaded `.exe` file

3. Create a new environment in Miniconda using the provided `gan-keras.yml` file using the following command:
    ```
    conda env create -f gan-keras.yml
    ```
    This will create a new environment named `gan-keras`
    
4. Activate the created environment,
    - In Linux, use the following command:
    ```
    source activate gan-keras
    ```
    - In Windows, use the following command:
    ```
    activate gan-keras
    ```

5. Start a Jupyter notebook server by using the following command:
    ```
    jupyter notebook
    ```


## gan-exploration Notebook

GAN experiment for MNIST dataset. Also contains classifier for MNIST that is used to label generated images. The labels are then filtered with certain threshold to separate high and low quality data. Generated images and labels that are above the given threshold (high quality) are then used as training data alongside the original dataset.

A new classifier is then trained with the combined data and compared with classifier that uses only the original dataset. The result shows **slight advantage for classifier that uses GAN-generated data for training**.


## gan-toy Notebook

GAN experiment for toy dataset to determine if GAN-generated data could help in improving a classifier's performance on certain dataset.

### Numerical dataset

2 numerical attributes that is randomly generated from the normal distribution.

Dataset summary
- 10,000 training data
- 1,000 generated data

Data distribution for training data
- x1 mean = 0.000965, std = 0.977644
- x2 mean = 4.967461, std = 2.029867

Data distribution for generated data
- x1 mean = 0.011561, std = 0.959017
- x2 mean = 4.785554, std = 2.037395

### Numerical Dataset + Label

2 numerical attributes that is randomly generated from the normal distribution with numerical label.

Dataset summary
- 10,000 training data
- 5,000 test data
- 1,000 generated data

Classifier comparison
- Mean Squared Error = 1.021194 (training data)
- Mean Squared Error = 1.010679 (training + generated data)

### Categorical dataset

2 categorical attributes (5 & 2 classes) that is randomly generated from the normal distribution and categorized based on the value.

Found out a way to generate categorical data (one-hot encoded) with GAN using Keras' Model functional API.
```python
def build_generator(self):
    noise = Input(shape=self.noise_shape)

    hidden_1 = Dense(18)(noise)
    hidden_1 = LeakyReLU(alpha=0.2)(hidden_1)
    hidden_1 = BatchNormalization(momentum=0.8)(hidden_1)
    
    hidden_2 = Dense(16)(hidden_1)
    hidden_2 = LeakyReLU(alpha=0.2)(hidden_2)
    hidden_2 = BatchNormalization(momentum=0.8)(hidden_2)
        
    # Branch 1 - Categorical (5)
    branch_1_hidden_1 = Dense(12)(hidden_2)
    branch_1_hidden_1 = LeakyReLU(alpha=0.2)(branch_1_hidden_1)
    branch_1_hidden_1 = BatchNormalization(momentum=0.8)(branch_1_hidden_1)
    
    branch_1_hidden_2 = Dense(7)(branch_1_hidden_1)
    branch_1_hidden_2 = LeakyReLU(alpha=0.2)(branch_1_hidden_2)
    branch_1_hidden_2 = BatchNormalization(momentum=0.8)(branch_1_hidden_2)
    
    branch_1_output = Dense(5, activation='softmax')(branch_1_hidden_2)

    # Branch 2 - Categorical (2)
    branch_2_hidden_1 = Dense(8)(hidden_2)
    branch_2_hidden_1 = LeakyReLU(alpha=0.2)(branch_2_hidden_1)
    branch_2_hidden_1 = BatchNormalization(momentum=0.8)(branch_2_hidden_1)

    branch_2_hidden_2 = Dense(4)(branch_2_hidden_1)
    branch_2_hidden_2 = LeakyReLU(alpha=0.2)(branch_2_hidden_2)
    branch_2_hidden_2 = BatchNormalization(momentum=0.8)(branch_2_hidden_2)
    
    branch_2_output = Dense(2, activation='softmax')(branch_2_hidden_2)

    merged_output = concatenate([branch_1_output, branch_2_output])

    return Model(inputs=noise, outputs=merged_output)

```

### Mixed dataset

2 categorical attributes (5 & 2 classes), and 2 numerical attributes.

Found out a way to generate both categorical and numerical values with GAN using Keras' Model functional API.
```python
def build_generator(self):
    noise = Input(shape=self.noise_shape)

    hidden_1 = Dense(18)(noise)
    hidden_1 = LeakyReLU(alpha=0.2)(hidden_1)
    hidden_1 = BatchNormalization(momentum=0.8)(hidden_1)
    
    hidden_2 = Dense(16)(hidden_1)
    hidden_2 = LeakyReLU(alpha=0.2)(hidden_2)
    hidden_2 = BatchNormalization(momentum=0.8)(hidden_2)
        
    # Branch 1 - Categorical (5)
    branch_1_hidden_1 = Dense(12)(hidden_2)
    branch_1_hidden_1 = LeakyReLU(alpha=0.2)(branch_1_hidden_1)
    branch_1_hidden_1 = BatchNormalization(momentum=0.8)(branch_1_hidden_1)
    
    branch_1_hidden_2 = Dense(7)(branch_1_hidden_1)
    branch_1_hidden_2 = LeakyReLU(alpha=0.2)(branch_1_hidden_2)
    branch_1_hidden_2 = BatchNormalization(momentum=0.8)(branch_1_hidden_2)
    
    branch_1_output = Dense(5, activation='softmax')(branch_1_hidden_2)

    .
    .
    .

    # Branch 3 - Numerical
    branch_3_hidden_1 = Dense(4)(hidden_2)
    branch_3_hidden_1 = LeakyReLU(alpha=0.2)(branch_3_hidden_1)
    branch_3_hidden_1 = BatchNormalization(momentum=0.8)(branch_3_hidden_1)
    
    branch_3_output = Dense(1)(branch_3_hidden_1)

    .
    .
    .

    merged_output = concatenate([branch_1_output, branch_2_output, branch_3_output, ...])

    return Model(inputs=noise, outputs=merged_output)

```

### Mixed Dataset + Label

2 categorical attributes (5 & 2 classes), and 2 numerical attributes with categorical label (3 classes).

Dataset summary
- 10,000 training data
- 5,000 test data
- 1,000 generated data

The generated data is labelled using previously trained classifier on training data with a threshold of 0.999. This means the classifier is at least 99.9% certain that the data is classified correctly. As the classifier has around 88-89% accuracy, a threshold of 99.9% should be reliable enough for data labelling. 

Classifier comparison
- Accuracy = 88.32% (training data)
- Accuracy = 88.66% (training + generated data)

## gan-variations Notebook

GAN and its variations throughout the years, written in chronological order from the first GAN paper to the latest. Only popular and/or significant GAN variations are included in the notebook.

Also contains a short explanation about some datasets mentioned in the papers.

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