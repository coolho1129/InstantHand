import external.handavatar.configs as handavatar_configs


def _cfg():
    return handavatar_configs.cfg


def _build_dataset_attrs(cfg):
    dataset_attrs = {}

    if cfg.category in ['handavatar'] and cfg.task == 'interhand':
        dataset_attrs.update({
            "interhand_train": {
                "dataset_path": f"data/InterHand/{cfg.interhand.fps}",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "interhand_test": {
                "dataset_path": f"data/InterHand/{cfg.interhand.fps}",
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": "image",
                "src_type": "interhand",
            },
        })

    if cfg.category in ['handavatar'] and cfg.task == 'hanco':
        dataset_attrs.update({
            "hanco_train": {
                "dataset_path": "data/HanCo",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "hanco_test": {
                "dataset_path": "data/HanCo",
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": "image",
                "src_type": "hanco",
            },
        })

    return dataset_attrs


class DatasetArgs(object):
    dataset_attrs = {}

    @staticmethod
    def get(name):
        attrs = _build_dataset_attrs(_cfg())
        if name not in attrs:
            raise KeyError(name)
        return attrs[name].copy()
