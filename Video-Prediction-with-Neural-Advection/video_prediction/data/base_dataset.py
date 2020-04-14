import os
import glob
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Base class for 
    """

    def __init__(self, opt, mode="train"):
        self.mode = mode
        self.opt = opt
        self.data_dir = os.path.join(opt.data_dir, self.mode)

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    @abstractmethod
    def len(self):
        return 0

    @abstractmethod
    def get_item(self):
        pass    

    @abstractmethod
    def print_info(self):
        pass


class TFDataset(BaseDataset):

    def __init__(self, 
                opt, mode="train"):
        BaseDataset.__init__(self, opt, mode=mode)
        
        self.filenames = glob.glob(os.path.join(self.data_dir, "*.tfrecord*"))

    @staticmethod
    def modify_commandline_options(parser):
        BaseDataset.modify_commandline_options(parser)

        parser.add_argument("--num_parallel_calls", type=int, default=4)
        parser.add_argument("--data_buffer_size", type=int, default=8 * 1024 * 1024)
        parser.add_argument("--shuffle_buffer_size", type=int, default=1024)
        parser.add_argument("--drop_remainder", type=int, default=1)

    def filter(self, serialized_example):   
        raise NotImplementedError

    def parse(self, serialized_example):
        raise NotImplementedError

    def make_dataset(self):
        raise NotImplementedError
  
