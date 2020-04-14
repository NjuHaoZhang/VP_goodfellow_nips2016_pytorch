import tensorflow as tf 
import tensorflow.compat.v1 as v1
from collections import OrderedDict
from video_prediction.models.base_model import BaseModel
from video_prediction.models.networks import define_net
from video_prediction.utils.losses import l2_loss
from video_prediction.utils.metrics import peak_signal_to_noise_ratio
from video_prediction.utils.tf_ops import add_summaries, IMAGE_SUMMARIES, SCALAR_SUMMARIES


class PixelAdvectionModel(BaseModel):
    """
    Implementing pixel advection model.

    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.net_forward = define_net("pixel_advection", opt=opt)

    @staticmethod
    def modify_commandline_options(parser):
        BaseModel.modify_commandline_options(parser)

        parser.add_argument("--context_len", type=int, default=2)
        parser.add_argument("--use_predict_frame", type=int, default=0)
        parser.add_argument("--schedule_k", type=float, default=900.0)

        parser.add_argument("--stp", type=int, default=0)
        parser.add_argument("--cdna", type=int, default=1)
        parser.add_argument("--dna", type=int, default=0)
        parser.add_argument("--dna_kernel_size", type=int, default=5)
        # Amount to use when lower bounding tensors
        parser.add_argument("--relu_shift", type=float, default=1e-12)
        parser.add_argument("--num_mask", type=int, default=10)

        parser.add_argument("--recon_loss_weight", type=float, default=1.0)
        parser.add_argument("--state_loss_weight", type=float, default=1e-4)

        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        return parser

    def set_input(self, model_input):
        self.model_input = model_input

    def build_graph(self):
        """
        Only apply self.xxx to some shared variables, since this model is shared by "train" and "val"
        """
        self.global_step = v1.train.get_or_create_global_step()
        # used to seperate each mode when computing metrics/adding summaries

        # split input by each time step
        input_dict = {}
        for k, v in self.model_input.items():
            v = tf.split(axis=1, num_or_size_splits=int(v.get_shape()[1]), value=v)
            input_dict[k] = [tf.squeeze(one_step) for one_step in v]

        # forward pass #
        with v1.variable_scope("forward"):
            gen_images, gen_r_states = self.forward(input_dict, mode="train")

        # compute loss in train/val mode # 
        # not including the context frames
        target_images = input_dict["image"][self.opt.context_len:]
        target_r_states = input_dict["r_state"][self.opt.context_len:]
        gen_images = gen_images[self.opt.context_len - 1:]
        gen_r_states = gen_r_states[self.opt.context_len -1:]
        with v1.variable_scope("forward_loss"):
            losses_dict = self.compute_losses(target_images, target_r_states, gen_images, gen_r_states)
    
        # compute metrics in train/val modes #
        with v1.variable_scope("forward_metric"):
            metrics_dict = self.compute_metrics(target_images, gen_images)

        # optimize parameters #
        with tf.name_scope("optimize_parameters"):
            vars_forward = v1.trainable_variables("forward")
            # currently no decay here     
            self.optimizer_forward = v1.train.AdamOptimizer(self.opt.lr, self.opt.beta1, self.opt.beta2)
            with tf.control_dependencies(v1.get_collection(v1.GraphKeys.UPDATE_OPS)):
                with tf.name_scope("compute_gradients_forward"):
                    gradvars_forward = self.optimizer_forward.compute_gradients(losses_dict["forward_loss"], var_list=vars_forward)
                with tf.name_scope("apply_gradients_forward"):
                    train_forward_op = self.optimizer_forward.apply_gradients(gradvars_forward)
            with tf.control_dependencies([train_forward_op]): 
                train_op = v1.assign_add(self.global_step, 1)       
        
        # add summaries # 
        with tf.name_scope("add_summary"):
            image_sum_op, scalar_sum_op = self.create_summaries(losses_dict, metrics_dict, {"target":tf.stack(target_images, axis=1)},\
                {"gen_image":tf.stack(gen_images, axis=1)}, mode="train")
            eval_image_sum_op, eval_scalar_sum_op = self.create_summaries(losses_dict, metrics_dict, {"target":tf.stack(target_images, axis=1)},\
                {"gen_image":tf.stack(gen_images, axis=1)}, mode="eval")        

        # store fetches # 
        fetch_dict = {}
        fetch_dict["train_op"] = train_op
        
        fetch_dict["global_step"] = self.global_step
        fetch_dict["losses"] = losses_dict
        fetch_dict["metrics"] = metrics_dict
        fetch_dict["scalar_sum_op"] = scalar_sum_op
        fetch_dict["image_sum_op"] = image_sum_op
        fetch_dict["eval_scalar_sum_op"] = eval_scalar_sum_op
        fetch_dict["eval_image_sum_op"] = eval_image_sum_op
        return fetch_dict

    def forward(self, input_dict, mode="train"):
        gen_images, gen_r_states = self.net_forward.call(input_dict, mode=mode)
        return gen_images, gen_r_states

    def compute_losses(self, target_images, target_r_states, gen_images, gen_r_states):
        """
        Computing losses. By default, computing the l2 losses of generated and target 
        images/states (without context frames).
        Parameters:
            target_images: list of target images 
            target_r_states: list of target states
            gen_images: list of generated images
            gen_r_states: list of generated states
        """
        recon_loss = l2_loss(gen_images, target_images)
        r_state_loss = l2_loss(gen_r_states, target_r_states)
        loss_forward = self.opt.recon_loss_weight * recon_loss + self.opt.state_loss_weight * r_state_loss

        losses_dict = {}
        losses_dict["recon_loss"] = recon_loss
        losses_dict["r_state_loss"] = r_state_loss
        losses_dict["forward_loss"] = loss_forward
        # time reduction
        for k, v in losses_dict.items():
            v /= len(gen_images)
        return losses_dict

    def compute_metrics(self, target_images, gen_images):
        """
        Computing metrics. By default, computing the peak_signal_to_noise_ratio of generated and target 
        images (without context frames).
        Parameters:
            target_images: list of target images 
            gen_images: list of generated images
        """        
        psnr = peak_signal_to_noise_ratio(target_images, gen_images)
        metrics_dict = {}
        metrics_dict["psnr_metric"] = psnr
        return metrics_dict

    def create_summaries(self, losses_dict, metrics_dict, targets, gen_images, mode="train"):
        prefix = mode

        # rename keys to avoid name conflict
        for one_dict in [losses_dict, metrics_dict, targets, gen_images]:
            for k, v in list(one_dict.items()):
                one_dict[prefix + "_" + k] = one_dict.pop(k)

        # add loss summaries, scalar
        with tf.name_scope("loss_summaries"):
            add_summaries(losses_dict)
        
        # add metric summaries, scalar
        with tf.name_scope("metric_summaries"):
            add_summaries(metrics_dict)
        
        # add target images, image
        with tf.name_scope("target_summaries"):
            add_summaries(targets)
        
        # add generated image, image
        with tf.name_scope("gen_image_summaries"):
            add_summaries(gen_images)

        image_sum_op = v1.summary.merge(v1.get_collection(IMAGE_SUMMARIES))
        scalar_sum_op = v1.summary.merge(v1.get_collection(SCALAR_SUMMARIES))

        return image_sum_op, scalar_sum_op