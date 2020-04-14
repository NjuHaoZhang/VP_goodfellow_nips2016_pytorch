import argparse
import os
from video_prediction.data import find_dataset_using_name
from video_prediction.models import find_model_using_name
from video_prediction.utils.general_ops import mkdir_given_options, print_options

class BaseOptions():
    """
    Base option class for train.py and test.py
    """
    def __init__(self):
        self.initialized = False
        # to be defined in either train_option.py or test_option.py
        self.isTrain = None        

    def initialize(self, parser):
        parser.add_argument('--data_dir', type=str, required=True, help="path to a data directory, containing train / val / test data.\
                                                                Should be a relative path")
        parser.add_argument('--exp_root', type=str, default="video_prediction/exp", help='dir to store experiment results')
        parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoint_name', type=str, help="(optional), name of pre-trained model")
        parser.add_argument('--verbose', type=int, default=0, help="1 to enable verbose mode, save more intermediate values")
        parser.add_argument("--steps_per_epoch", type=int, help="number of step per epoch")
        parser.add_argument("--max_epoch", type=int, help="max running epochs")
        parser.add_argument("--max_step", type=int, help="max running steps")

        # dataset options
        parser.add_argument('--dataset_type', type=str, required=True, help='type of dataset, used in __init__.py in data')
        parser.add_argument('--batch_size', type=int, required=True, help="batch size of dataset")
        # model options
        parser.add_argument('--model_type', type=str, required=True, help='type of model, used in __init__.py in models')
        self.initialized = True
        return parser

    def gather_options(self):
        # check if it has been initialized
        if not self.initialized:  
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args() 

        # get dataset-specific options
        dataset_option_setter = find_dataset_using_name(opt.dataset_type).modify_commandline_options
        parser = dataset_option_setter(parser)
        # parse again with new defaults
        opt, _ = parser.parse_known_args()  

        # get model-specific options
        model_option_setter = find_model_using_name(opt.model_type).modify_commandline_options
        parser = model_option_setter(parser)
        # parse again with new defaults
        opt, _ = parser.parse_known_args()  
        
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse_main(self):
        opt = self.gather_options()                                                         # get all options
        opt.isTrain = self.isTrain      # set experiment mode
        opt.exp_dir = os.path.join(opt.exp_root, opt.exp_name)
        opt.sum_dir = os.path.join(opt.exp_dir, "summary")
        if self.isTrain:
            opt.checkpoint_dir = os.path.join(opt.exp_dir, "checkpoint/train")
        if os.path.exists(opt.sum_dir): # clean up former summaries
            os.system("rm -rf %s" %opt.sum_dir)
        
        mkdir_given_options(opt)                                                            # make dirs if not exist
        print_options(opt)                                                                  # save/print all options

        self.opt = opt
        return self.opt

