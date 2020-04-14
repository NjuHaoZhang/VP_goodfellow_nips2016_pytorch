import sys
sys.path.append("../..")
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as v1
import random
import os
import re
from collections import OrderedDict
#from google.protobuf.json_format import MessageToDict
from video_prediction.data.base_dataset import TFDataset
#tf.enable_eager_execution()

NUM_TRAIN = 51531
NUM_VAL = 111

class GooglePushDataset(TFDataset):

    def __init__(self, opt, mode="train", init="one_shot"):
        TFDataset.__init__(self, opt, mode=mode)
        # get file infos
        self.sequence_len = opt.sequence_len
        self.use_state = opt.use_state
        self.state_dim = opt.state_dim
        self.original_height = opt.original_height
        self.original_width = opt.original_width
        self.image_height = opt.image_height
        self.image_width = opt.image_width
        self.color_ch = opt.color_ch

        self.dataset = self.make_dataset()

        if init == "one_shot":
            # equal to self.dataset.make_one_shot_dataset()
            self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)  
            self.initializer = None
        elif init == "initialize":
            self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
            self.initializer = self.iterator.initializer
        else:
            raise ValueError("Invalid dataset initialization type [%s]" %init)
        # get the string handle of this one-shot iterator
        # see example in https://www.tensorflow.org/api_docs/python/tf/data/Iterator#from_string_handle
        self.handle = self.iterator.string_handle()      

    @staticmethod
    def modify_commandline_options(parser):
        TFDataset.modify_commandline_options(parser)

        parser.add_argument("--sequence_len", type=int, default=10)
        parser.add_argument("--use_state", type=int, default=1)
        parser.add_argument("--state_dim", type=int, default=5)
        parser.add_argument("--original_height", type=int, default=512)
        parser.add_argument("--original_width", type=int, default=640)
        parser.add_argument("--image_height", type=int, default=64)
        parser.add_argument("--image_width", type=int, default=64)
        parser.add_argument("--color_ch", type=int, default=3)

        return parser

    def _test_dataset(self):    
        """ test using eager mode """ 
        
        assert(tf.executing_eagerly())
        print("Using eager mode: ", tf.executing_eagerly())
        count = 1
        for x in self.dataset:
            print(count)
            count += 1
            print(x['image'])
            print(x["action"])
            print(x["r_state"])
            assert(0)

    def len(self):
        if self.mode == "train":
            return NUM_TRAIN
        elif self.mode == "val":
            return NUM_VAL

    def get_item(self):
        return self.handle

    def print_info(self):
        print("="*50)
        print("dataset [%s] was created in mode[%s], data # [%d]" % (self.opt.dataset_type, 
                                                                    self.mode, self.len()))
        print("Model input information:")
        one_data = self.iterator.get_next()
        for key, value in one_data.items():
            print("\tkey: ", key, "\tvalue shape ", value)
            
        print("="*50)
    
    def filter(self, serialized_example):
        return tf.convert_to_tensor(True)

    def parse(self, serialized_example):
        # images, r_states, actions are stored by steps
        image_seq, r_state_seq, action_seq = [], [], []
        for i in range(self.sequence_len):
            features = dict()
            image_name = 'move/' + str(i) + '/image/encoded'
            features[image_name] = tf.io.FixedLenFeature([1], tf.string)
            if self.use_state:
                action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
                r_state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'                
                features[action_name] = tf.io.FixedLenFeature([self.state_dim], tf.float32)
                features[r_state_name] = tf.io.FixedLenFeature([self.state_dim], tf.float32)

            features = tf.io.parse_single_example(serialized_example, features=features)
            # decode, crop, cast
            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=self.color_ch)
            image = tf.reshape(image, [self.original_height, self.original_width, self.color_ch])

            if self.image_height != self.image_width:
                raise ValueError('Unequal height and width unsupported')

            crop_size = min(self.original_height, self.original_width)
            image = tf.image.resize_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, self.color_ch])
            image = v1.image.resize_bicubic(image, [self.image_height, self.image_width])
            image = tf.cast(image, tf.float32) / 255.0
            image_seq.append(image)

            if self.use_state:
                r_state = tf.reshape(features[r_state_name], shape=[1, self.state_dim])
                r_state_seq.append(r_state)
                action = tf.reshape(features[action_name], shape=[1, self.state_dim])
                action_seq.append(action)            
        image_seq = tf.concat(image_seq, axis=0)

        if self.use_state:
            r_state_seq = tf.concat(r_state_seq, axis=0)
            action_seq = tf.concat(action_seq, axis=0)  
        else:
            # return zero r_state and action
            r_state_seq = tf.zeros([self.sequence_len, self.state_dim])
            action_seq = tf.zeros([self.sequence_len, self.state_dim])    

        ret_seq = OrderedDict()
        ret_seq["image"] = image_seq
        ret_seq["r_state"] = r_state_seq
        ret_seq["action"] = action_seq
        return ret_seq

    def make_dataset(self):
        filenames = self.filenames
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=self.opt.data_buffer_size)
        # shuffle
        if self.mode == "train":
            dataset = dataset.shuffle(self.opt.shuffle_buffer_size, seed=None)
        # filter function
        dataset = dataset.filter(self.filter)
        # map function
        dataset = dataset.map(self.parse, num_parallel_calls=self.opt.num_parallel_calls if self.opt.num_parallel_calls \
            else tf.data.experimental.AUTOTUNE)
        # batch 
        dataset = dataset.batch(self.opt.batch_size, drop_remainder=bool(self.opt.drop_remainder))
        # repeat
        dataset = dataset.repeat(self.opt.max_epoch)       
        # prefetch 
        dataset = dataset.prefetch(self.opt.batch_size)
        return dataset        
 
