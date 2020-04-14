
from video_prediction.data.google_push_dataset import GooglePushDataset

def find_dataset_using_name(dataset_type):
	if dataset_type == "google_push":
		return GooglePushDataset
	else:
		raise NotImplementedError

def create_dataset(opt, mode="train", init="one_shot"):
	dataset = find_dataset_using_name(opt.dataset_type)(opt, mode=mode, init=init)
	dataset.print_info()
	return dataset
    
