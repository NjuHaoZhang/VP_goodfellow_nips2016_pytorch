import os
from collections import OrderedDict

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdir_given_options(opt):
    for k, v in sorted(vars(opt).items()):
        if (str(k).endswith("_dir", len(str(k)) - 4, len(str(k)))):     # for those dirs                            # convert to absolution path
            mkdir(str(v))       
    return opt       

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(opt.exp_dir, 'opt_train.txt') if opt.isTrain else os.path.join(opt.exp_dir, 'opt_test.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def print_current_loss(log_file, epoch, epoch_step, steps_per_epoch, 
        epoch_time, remain_time, images_per_sec, report):

    message = '|epoch: %d| progress: %d/%d| image/sec %0.1f | epoch_time: %.3f m| remain: %.3f m\n' % (epoch, epoch_step, 
    steps_per_epoch, images_per_sec, epoch_time/60, remain_time/60)
    for k, v in report.items():
        message +='%s: %.6f   ' % (k, v)
    print(message)

    with open(log_file, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

def tuple_list_to_orderdict(x):
    assert(isinstance(x, list))
    ret_dict = OrderedDict()
    for i in range(len(x)):
        ret_dict[x[i][0]] = x[i][1]
    return ret_dict