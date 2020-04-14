# Video Prediction with Neural Advection

*A TensorFlow implementation of the models described in [Unsupervised Learning for Physical Interaction through Video Prediction (Finn et al., 2016)](https://arxiv.org/abs/1605.07157).*

This implementation is based on the TensorFlow official implementation at [here](https://github.com/tensorflow/models/tree/master/research/video_prediction), while making it compatible with TensorFlow 1.14 version. Basically, it removed depreciated tf.contrib.slim and use mostly tf.keras.layers instead.

## Requirements
* Linux or macOS
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN
* TensorFlow 1.14 version (the newest tf.1x version)

## Data
The data used to train this model is located
[here](https://sites.google.com/site/brainrobotdata/home/push-dataset).

To download the robot data, run the following.
```shell
./download_data.sh
```
Data should be placed in  `video_prediction/dataset` folder.

## Training the model

To train the model, run the prediction_train.py file.
```shell
python train.py
```

