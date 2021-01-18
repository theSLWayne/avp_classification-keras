# Aliens vs Predator Classification

This is a Deep Learning CNN model created to classfiy aliens and predators images. The model is trained on and evaluated using images from the public dataset [Alien vs. Predator images](https://www.kaggle.com/pmigdal/alien-vs-predator-images) on [Kaggle](www.kaggle.com).

## Files

- *create_model.py*: The python script that creates, trains and saves the model.
- *make_preds.py*: The python script that loads the model and evaluate it using test dataset.
- *tf_model.h5*: Saved model

## Model

The model takes preprocessed images from the dataset. The model contains 2 Max Pooling layers, 2 Conolution layers and one fully-connected layer.  

## Callback

In order to prevent the model from overfitting, a callback is implemented. Callback stops training the model whenever it reaches to 80% accuracy.