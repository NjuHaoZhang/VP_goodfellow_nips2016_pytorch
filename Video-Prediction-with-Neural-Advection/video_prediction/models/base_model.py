import os
import tensorflow as tf
import math
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.mode = mode 

    @staticmethod
    def modify_commandline_options(parser):
        return parser
    
    @abstractmethod
    def set_input(self, model_input):
        raise NotImplementedError

    @abstractmethod
    def build_graph(self, mode="train"):
        raise NotImplementedError

#     @abstractmethod
#     def optimize_parameters(self):
#         pass

#     @abstractmethod
#     def add_summaries(self):
#         pass    

#     @abstractmethod
#     def add_fetches(self, opt):
#         pass    
