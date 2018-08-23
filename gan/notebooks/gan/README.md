# GAN

## gen-data

Contains generated data from `gan-toy` notebook. Data in npy format.
- numerical/categorical/mix indicates the data type (mix is combination of numerical and categorical)
- label indicates the data has label
- original/generated indicates how the data is created (original is generated manually, generated is generated with GAN)
- train/test indicates which data is used for training or testing
- x/y indicates the data

## gen-images

Contains generated MNIST images from `gan-exploration` notebook.
- mse/bce indicates the model's loss function when generating the data (mse for mean squared error, bce for binary crossentropy)
- x/y indicates the data (x for images(28,28,1), y for labels)

## images

Empty folder where image samples from `gan-exploration` notebook will be saved while training GAN for MNIST. 

## images-bce

Contains generated MNIST images from `gan-exploration` notebook while training GAN for MNIST with BCE loss.

## images-mse

Contains generated MNIST images from `gan-exploration` notebook while training GAN for MNIST with MSE loss.

## logs

Contains training logs from both `gan-exploration` and `gan-toy` notebook.

The format is `{Discriminator's Loss},{Generator's Loss}`.

## models

Empty folder where models from both `gan-exploration` and `gan-toy` will be saved during or after training models (classifiers and GANs)