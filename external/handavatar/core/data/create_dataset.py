"""Dataset / DataLoader factory for the HandAvatar-format evaluation datasets.

Like dataset_args.py, the config is read at call time (see the note there).
"""
import imp
import time

import numpy as np
import torch

import external.handavatar.configs as handavatar_configs
from .dataset_args import DatasetArgs


def _cfg():
    return handavatar_configs.cfg


def _query_dataset(data_type):
    module = _cfg()[data_type].dataset_module
    if module.startswith("handavatar."):
        module = "external." + module
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Dataset


def create_dataset(data_type='train', subject=None):
    cfg = _cfg()
    args = DatasetArgs.get(cfg[data_type].dataset)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    args['data_type'] = data_type
    if data_type == 'progress':
        if cfg.progress.get('skip', -1) > 0:
            args['skip'] = cfg.progress.skip
            args['maxframes'] = -1
        else:
            args['skip'] = 16
            args['maxframes'] = 16

    if data_type in ['freeview', 'tpose', 'freepose', 'infer']:
        args['skip'] = cfg[data_type].get('skip', 1)
    if data_type in ['infer']:
        args['subject'] = subject

    dataset = _query_dataset(data_type)
    return dataset(**args)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def create_dataloader(data_type='train', **kwargs):
    cfg_node = _cfg()[data_type]
    dataset = create_dataset(data_type=data_type, **kwargs)
    return torch.utils.data.DataLoader(dataset=dataset,
                                       batch_size=cfg_node.batch_size,
                                       shuffle=cfg_node.shuffle,
                                       drop_last=cfg_node.drop_last,
                                       num_workers=cfg_node.num_workers,
                                       worker_init_fn=_worker_init_fn)
