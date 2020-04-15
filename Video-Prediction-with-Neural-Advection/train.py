'''
@Author: haozhang
@Date: 2020-04-15 00:35:05
@LastEditTime: 2020-04-15 22:41:48
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \VP_goodfellow_nips2016_pytorch\Video-Prediction-with-Neural-Advection\train.py
'''

import sys 
sys.path.append('.')

import tensorflow as tf
from tensorflow.compat import v1 as v1
import time
import math
import os
from collections import OrderedDict

from video_prediction.data import create_dataset
from video_prediction.models import create_model
from video_prediction.options.train_options import TrainOptions
from video_prediction.utils.general_ops import print_current_loss

if __name__ == "__main__":

    opt = TrainOptions().parse_main()                                           # get options

    train_dataset = create_dataset(opt, mode="train", init="one_shot")          # create train and val dataset
    train_handle = train_dataset.get_item()         
    val_dataset = create_dataset(opt, mode="val", init="initialize")             
    val_handle = val_dataset.get_item()
    opt.steps_per_epoch = int(math.ceil(train_dataset.len() / opt.batch_size))
    opt.max_steps = opt.steps_per_epoch * opt.max_epoch

    dummy_handle = v1.placeholder(tf.string, shape=[])                # placeholder for string handle
    iterator = v1.data.Iterator.from_string_handle(dummy_handle,      # main iterator suitable for train and val handle
        v1.data.get_output_types(train_dataset.dataset), 
        v1.data.get_output_shapes(train_dataset.dataset))
    model_input = iterator.get_next()                                           # model input

    model = create_model(opt)                                                   # create model
    model.set_input(model_input)                                                # parse model input  
    model_fetch = model.build_graph() 

    train_fetch_keys = ["train_op", "global_step", "losses", "metrics", "scalar_sum_op", "image_sum_op"] 
    val_fetch_keys = ["global_step", "losses", "metrics", "eval_scalar_sum_op", "eval_image_sum_op"] 
    train_fetch = OrderedDict()
    for key in train_fetch_keys:
        train_fetch[key] = model_fetch[key]
    val_fetch = OrderedDict()
    for key in val_fetch_keys:
        val_fetch[key] = model_fetch[key]
                
    with v1.Session() as sess:
        summary_writer = v1.summary.FileWriter(opt.sum_dir, sess.graph)
        sess.run(v1.global_variables_initializer())
        for e in range(10):
            for i in range(opt.steps_per_epoch):
                res = sess.run(train_fetch, feed_dict={dummy_handle:sess.run(train_handle)})
                summary_writer.add_summary(res["scalar_sum_op"], res["global_step"])
                summary_writer.add_summary(res["image_sum_op"], res["global_step"])
            sess.run(val_dataset.initializer)
            for i in range(200):
                res = sess.run(val_fetch, feed_dict={dummy_handle:sess.run(val_handle)})
                summary_writer.add_summary(res["eval_scalar_sum_op"], res["global_step"])
                summary_writer.add_summary(res["eval_image_sum_op"], res["global_step"])