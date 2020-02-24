#-*- coding: utf-8 -*-
import argparse
import numpy as np

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def define_args_parser():
    parser = argparse.ArgumentParser(description='Benchmark settings.')
    return parser

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--learning_rate', '-l', default=0.05, type=float, help='The learning rate')
net_arg.add_argument('--kernel_size', '-k', type=int, default=4, help='The kernel size of each conv layer')
net_arg.add_argument('--depth', '-d', type=int, default=2, help='Num of conv layers before sum pooling')
net_arg.add_argument('--width', '-w', type=int, default=4, help='Num of output channels in eachconv layer')
net_arg.add_argument('--model_name', '-m', type=str, choices=["rbm","drbm","dnn","ar1", "ar2","my_rbm"], default='drbm', help='Model architecture')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--pb_type', type=str, choices=["maxcut", "spinglass"], default="maxcut", help='The problem type')
data_arg.add_argument('--batch_size', '-b', type=int, default=128, help='The batch size in each iteration')
data_arg.add_argument('--input_size', '-i', nargs="+", type=int, default=20, help='Number of spins in the input')
data_arg.add_argument('--pyket_num_of_chains', type=int, default=20, help='Num of parralel mcmc in flowket')
data_arg.add_argument('--num_of_iterations', type=int, default=500, help='Num of iterations to benchmark')

# Train
train_arg = add_argument_group('Training')
train_arg.add_argument('--epochs', type=int, default=200)
train_arg.add_argument('--use_cholesky', action='store_true', help='use cholesky solver in SR')
train_arg.add_argument('--use_iterative',action='store_true', help='use iterative solver in SR')
train_arg.add_argument('--pyket_on_cpu', '-cpu',  action='store_true', help='force running flowket on cpu')
train_arg.add_argument('--optimizer', choices=["adam","sr","sgd"], default="adam", help='The optimizer for training')
train_arg.add_argument('--fast_jacobian', action='store_true', help='use flowket custom code for jacobian (still have bugs)')
train_arg.add_argument('--no_pfor', action='store_true', help="don't use tensorflow pfor")
train_arg.add_argument('---optimizer', type=str, default='sgd')
train_arg.add_argument('--scheduler', type=str, default='normal')

# Evaluation
eval_arg = add_argument_group('Evaluation')
eval_arg.add_argument('--save_model_name', type=str, default="robust_model.pth")
eval_arg.add_argument('--model_load_path', type=str, default="./pretrained_models/model.pth")

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--framework', '-fr', type=str, choices=['flowket', 'netket'], default='flowket')
misc_arg.add_argument('--dir', type=str, default='')
misc_arg.add_argument('--data_path', type=str, default='datasets')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', '-r', type=int, default=499, help='Randomization seed')
misc_arg.add_argument('--num_workers', type=int, default=4)
misc_arg.add_argument('--log_interval', type=int, default=10)


def get_config():
    cf, unparsed = parser.parse_known_args()
    if len(cf.input_size) == 1:
        cf.input_size = (cf.input_size[0],)
    elif len(cf.input_size) == 2:
        cf.input_size = (cf.input_size[0],cf.input_size[1])
    else:
        raise("input dimension must be 1 or 2")
    return cf, unparsed
