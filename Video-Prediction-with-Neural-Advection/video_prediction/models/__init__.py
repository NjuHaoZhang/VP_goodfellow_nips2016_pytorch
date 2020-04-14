from video_prediction.models.pixel_advection_model import PixelAdvectionModel
from video_prediction.models.networks import define_net

def find_model_using_name(model_type):
    if model_type == 'pixel_advection':
        return PixelAdvectionModel
    else:
        raise NotImplementedError

def create_model(opt):
    model = find_model_using_name(opt.model_type)(opt)
    return model
