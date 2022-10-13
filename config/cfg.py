import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-s', '--seed', type=int, help='seed')
    parser.add_argument('-a', '--arch', type=str, help='model name')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('--train_batch_size', type=int, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, help='test batch size')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('-c', '--config_file', type=str, help='config filename')
    parser.add_argument('-t', '--title', type=str, help='graph title')

    args = parser.parse_args()
    if args.config_file is None:
        # print('===========')
        args.config_file = f'{args.arch}.yaml'
    if args.model_name is None:
        args.model_name = args.arch
    
    # print(args.model_name)#########

    args_dict = vars(args)
    args_list = []
    for key, value in zip(args_dict.keys(), args_dict.values()):
        if value is not None:
            args_list.append(key)
            args_list.append(value)

    yaml_file = os.path.join(args.config_path, args.config_file)
    # print(yaml_file)
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(args_list)
    cfg.download = not os.path.exists(os.path.join(cfg.dataset_path, cfg.dataset_name))
    cfg.log_path = f'{cfg.log_path}/{cfg.model_name}'
    cfg.freeze()

    # Print cfg
    # for k, v in cfg.items():
    #     print(f'{k}: {v}')

    return cfg

cfg = parse()
