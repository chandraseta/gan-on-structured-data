# Generative Adversarial Network (GAN)

GAN is a concept introduced by Ian Goodfellow in 2014 for new generative model. The model is composed of two networks, a generator and a discriminator. The generator will create fake data and the discriminator will try to guess if a data is fake or real. GAN will reach global optimum when the generator's data distribution is equal to the real data distribution.

Throughout the years, GAN has been extensively used on unstructured dataset, mainly images. The goal of this experiment is to **determine if GAN could be used on structured dataset**. In theory, GAN-generated data could be used as training data to increase the performance of a classifier. This could help immensely in training on imbalanced dataset.

## deep-learning

Contains exploration on deep learning before diving deeper about GAN.

## gan

Contains exploration about GAN on MNIST and structured (numerical + categorical) dataset.