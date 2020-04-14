import tensorflow as tf
import tensorflow.compat.v1 as v1
import tensorflow.compat.v1.keras as keras
import tensorflow.compat.v1.keras.layers as Layers
from video_prediction.models.building_blocks import basic_conv_lstm_cell
from video_prediction.utils.general_ops import tuple_list_to_orderdict
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np 

###############################################################################
# Helper Functions
###############################################################################
def define_net(net_name, opt=None):
    """
    Main function for defining a network.
    Parameters:
        opt (option class) -- stores all options and parameters
        net_name (str)     -- network name
    """
    net = None
    if net_name == "pixel_advection":
        net = PixelAdvectionNetwork(opt)
    else:
        raise ValueError("net [%s] is not found." %net_name)

    return net

###############################################################################
# Network Classes
###############################################################################
class BaseNetwork(ABC):
    """
    Base class for network. Basically, each subset should implement at least "call" function.
    Parameters:
        opt (option class) -- stores all options and parameters    
    """
    def __init__(self, opt):
        self.opt = opt
        self.verbose = opt.verbose          # used to save more intermediate results

    @abstractmethod
    def call(self, *args, **kwargs):
        pass

class PixelAdvectionNetwork(BaseNetwork):
    """
    Implementing pixel advection network with one of dna, cdna and stp. You may choose to concatenate
    generated state in the middle.
    """
    def __init__(self, opt):
        BaseNetwork.__init__(self, opt)    

        # input sample
        self.k = opt.schedule_k   
        self.use_predict_frame = opt.use_predict_frame
        self.context_len = opt.context_len
        self.use_state = opt.use_state   
        # transform kernels
        self.dna = opt.dna
        self.cdna = opt.cdna 
        self.stp = opt.stp
        self.dna_kernel_size = opt.dna_kernel_size
        self.relu_shift = opt.relu_shift
        # mask 
        self.num_mask = opt.num_mask
        # channel size of Unet layers
        self.layer_ch_specs = [32, 32, 32, 64, 64, 128, 64, 32]
        self.layer_spatial_specs = [32, 32, 16, 16, 8, 16, 32]

        if self.stp + self.cdna + self.dna != 1:
            raise ValueError('More than one, or no network option specified.')
    
    def call(self, inputs, mode="train"):
        """
        Forward function for pixel advection network
        Parameters:
            inputs:         input dictionary including "image", "r_state" and "action"
              mode:         specify training or validating/testing
        Return:
            gen_images:     list of generated images
            gen_states:     list of generated states
        """
        ##### preparations #####
        # get dimensions/global steps
        global_step = tf.cast(v1.train.get_or_create_global_step(), tf.float32)  
        isTrain = True if mode == "train" else False
        batch_size, image_height, image_width, color_ch = inputs["image"][0].get_shape().as_list()[0:4]
        state_dim = inputs["r_state"][0].get_shape().as_list()[1]

        # placeholder for generated robot states and images
        gen_states, gen_images = [], []

        # initial r state will use ground truth
        current_r_state = inputs["r_state"][0]  

        # placeholder for conv-lstm states
        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        # get number of ground truth images used for each mini-batch
        num_ground_truth = tf.cast(
            tf.round(tf.cast(batch_size, tf.float32) * (self.k / (self.k + tf.exp(global_step / self.k)))), tf.int32)

        ###### begin time-step loop (total_len - 1 steps) ######
        for image, action in zip(inputs["image"][:-1], inputs["action"][:-1]):  

            ##### sampling and updating values #####
            # reuse parameters after the first step
            reuse = None if not bool(gen_images) else True

            # warm start(use ground truth frames) in first context_len steps
            done_warm_start = len(gen_images) > self.context_len - 1

            # if using context frames (during warm start), always use ground truth input
            # else, if not explicitly specified by "use_predict_frame", choose to use generated image 
            # or ground truth input based on sampling function
            if self.use_predict_frame and done_warm_start:
                current_image = gen_images[-1]
            elif done_warm_start:
                current_image = self.scheduled_sample(image, gen_images[-1], batch_size,
                                      num_ground_truth)
            else:
                current_image = image
        
            # concat state and action, always use ground truth action, but use current state
            current_state_action = tf.concat([action, current_r_state], axis=1)

            ##### begin U-net #####
            # 1th conv
            with v1.variable_scope("conv1", reuse=reuse):
                enc0 = Layers.Conv2D(self.layer_ch_specs[0], kernel_size=(5, 5), strides=(2, 2), padding="same")(current_image)
                enc0 = Layers.LayerNormalization()(enc0)

            # 1th conv lstm
            with v1.variable_scope("conv_lstm1", reuse=reuse):
                hidden1, lstm_state1 = basic_conv_lstm_cell(enc0, lstm_state1, self.layer_ch_specs[1])
                hidden1 = Layers.LayerNormalization()(hidden1)

            # 2th conv lstm
            with v1.variable_scope("conv_lstm2", reuse=reuse):
                hidden2, lstm_state2 = basic_conv_lstm_cell(hidden1, lstm_state2, self.layer_ch_specs[2])
                hidden2 = Layers.LayerNormalization()(hidden2)
                enc1 = Layers.Conv2D(self.layer_ch_specs[2], kernel_size=(3, 3), strides=(2, 2), padding="same")\
                            (hidden2)
            # 3th conv lstm
            with v1.variable_scope("conv_lstm3", reuse=reuse):
                hidden3, lstm_state3 = basic_conv_lstm_cell(enc1, lstm_state3, self.layer_ch_specs[3])
                hidden3 = Layers.LayerNormalization()(hidden3) 

            # 4th conv lstm
            with v1.variable_scope("conv_lstm4", reuse=reuse):
                hidden4, lstm_state4 = basic_conv_lstm_cell(hidden3, lstm_state4, self.layer_ch_specs[4])
                hidden4 = Layers.LayerNormalization()(hidden4)  
                enc2 = Layers.Conv2D(self.layer_ch_specs[4], kernel_size=(3, 3), strides=(2, 2), padding="same")\
                            (hidden4)  
                # Pass in state and action.
                smear = tf.reshape(
                    current_state_action, [batch_size, 1, 1, state_dim * 2])                             
                smear = tf.tile(
                    smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])

                if self.use_state:
                    enc2 = tf.concat(axis=3, values=[enc2, smear])
                enc3 = Layers.Conv2D(self.layer_ch_specs[4], kernel_size=(1, 1), strides=(1, 1), padding="same")\
                            (enc2)

            # 5th conv lstm
            with v1.variable_scope("conv_lstm5", reuse=reuse):
                hidden5, lstm_state5 = basic_conv_lstm_cell(enc3, lstm_state5, self.layer_ch_specs[5])
                hidden5 = Layers.LayerNormalization()(hidden5)              
                enc4 = Layers.Conv2DTranspose(self.layer_ch_specs[5], kernel_size=(3, 3), strides=(2, 2), padding="same")\
                    (hidden5)

            # 6th conv lstm
            with v1.variable_scope("conv_lstm6", reuse=reuse):
                hidden6, lstm_state6 = basic_conv_lstm_cell(enc4, lstm_state6, self.layer_ch_specs[6])
                hidden6 = Layers.LayerNormalization()(hidden6)                  
                # Skip connection.
                hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16
                enc5 = Layers.Conv2DTranspose(self.layer_ch_specs[6], kernel_size=(3, 3), strides=(2, 2), padding="same")\
                    (hidden6)                

            # 7th conv lstm
            with v1.variable_scope("conv_lstm7", reuse=reuse):
                hidden7, lstm_state7 = basic_conv_lstm_cell(enc5, lstm_state7, self.layer_ch_specs[7]) # 32x32
                hidden7 = Layers.LayerNormalization()(hidden7)      

                # Skip connection.
                hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32
                enc6 = Layers.Conv2DTranspose(self.layer_ch_specs[7], kernel_size=(3, 3), strides=(2, 2), padding="same")\
                    (hidden7)    
                enc6 = Layers.LayerNormalization()(enc6)      

            ###### motion transform part #####
            # dna
            if self.dna:
                from video_prediction.models.building_blocks import dna_transformation
                with v1.variable_scope("dna", reuse=reuse):
                    if self.num_mask != 1:
                        raise ValueError('Only one mask is supported for DNA model.')
                    dna_input = Layers.Conv2DTranspose(self.dna_kernel_size ** 2, kernel_size=(1, 1), strides=(1, 1),\
                        padding="same")(enc6)
                    transformed = [dna_transformation(current_image, dna_input, dna_kernel_size=self.dna_kernel_size, \
                        relu_shift=self.relu_shift)]
            # cdna
            elif self.cdna:
                from video_prediction.models.building_blocks import cdna_transformation
                with v1.variable_scope("cdna", reuse=reuse):
                    last_hidden_input = Layers.Conv2DTranspose(color_ch, kernel_size=(1, 1), strides=(1, 1),\
                        padding="same")(enc6)
                    transformed = [keras.activations.sigmoid(last_hidden_input)]
                    cdna_input = tf.reshape(hidden5, [batch_size, -1])
                    transformed += cdna_transformation(current_image, cdna_input, num_masks=self.num_mask, \
                        color_channels=color_ch, dna_kernel_size=self.dna_kernel_size, relu_shift=self.relu_shift)  
            # stp      
            elif self.stp:
                assert(0)
                from video_prediction.models.building_blocks import stp_transformation
                with v1.variable_scope("stp", reuse=reuse):
                    last_hidden_input = Layers.Conv2DTranspose(color_ch, kernel_size=(1, 1), strides=(1, 1),\
                        padding="same")(enc6)
                    transformed = [keras.activations.sigmoid(last_hidden_input)]
                    stp_input = tf.reshape(hidden5, [batch_size, -1])
                    stp_input = Layers.Dense(100)(stp_input)
                    transformed += stp_transformation(current_image, stp_input, self.num_mask)   
            
            # compute mask
            with v1.variable_scope("mask", reuse=reuse):
                mask = Layers.Conv2DTranspose(self.num_mask + 1, kernel_size=(1, 1), strides=(1, 1), padding="same")\
                    (enc6)
                mask = tf.reshape(
                    tf.nn.softmax(tf.reshape(mask, [-1, self.num_mask + 1])),
                    [batch_size, image_height, image_width, self.num_mask + 1])       
                #layers.append(("softmax_mask", mask))     
                mask_list = tf.split(axis=3, num_or_size_splits=self.num_mask + 1, value=mask)

            # mask output
            # first mask applies to current_image
            new_gen_image = mask_list[0] * current_image
            for layer, mask in zip(transformed, mask_list[1:]):
                new_gen_image += layer * mask    
            
            gen_images.append(new_gen_image)

            ###### compute new r state #####
            new_gen_r_state = Layers.Dense(state_dim)(current_state_action)   
            gen_states.append(new_gen_r_state)
            # update current state
            current_r_state = new_gen_r_state

        return gen_images, gen_states
                
    def scheduled_sample(self, ground_truth_x, generated_x, batch_size, num_ground_truth):
        """
        Sample batch with specified mix of ground truth and generated data points.
        Parameters:
            ground_truth_x: tensor of ground-truth data points.
            generated_x: tensor of generated data points.
            batch_size: batch size
            num_ground_truth: number of ground-truth examples to include in batch.
        Returns:
            New batch with num_ground_truth sampled from ground_truth_x and the rest
            from generated_x.
        """
        idx = tf.random.shuffle(tf.range(int(batch_size)))
        ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
        generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

        ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
        generated_examps = tf.gather(generated_x, generated_idx)
        return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                                [ground_truth_examps, generated_examps])