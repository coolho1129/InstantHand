"""Dataset arguments for the HandAvatar-format evaluation datasets (InterHand2.6M, HanCo).

`external.handavatar.configs.cfg` is read when `DatasetArgs.get()` is called, not at import:
the eval scripts build or replace the config after this module may already have been
imported, so the paths and keyfilters must follow the config that is active at call time.

Paths are relative to the repository root (run the scripts from there).
"""
import external.handavatar.configs as handavatar_configs


def _cfg():
    return handavatar_configs.cfg


def _build_dataset_attrs(cfg):
    dataset_attrs = {}

    if cfg.category in ['handavatar'] and cfg.task == 'interhand':
        dataset_attrs.update({
            "interhand_train": {
                "dataset_path": f"data/eval/InterHand/{cfg.interhand.fps}",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "interhand_test": {
                "dataset_path": f"data/eval/InterHand/{cfg.interhand.fps}",
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": "image",
                "src_type": "interhand",
            },
        })

    if cfg.category in ['handavatar'] and cfg.task == 'hanco':
        dataset_attrs.update({
            "hanco_train": {
                "dataset_path": "data/eval/HanCo",
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "hanco_test": {
                "dataset_path": "data/eval/HanCo",
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": "image",
                "src_type": "hanco",
            },
        })

    return dataset_attrs


class DatasetArgs(object):

    @staticmethod
    def get(name):
        attrs = _build_dataset_attrs(_cfg())
        if name not in attrs:
            raise KeyError(f"{name} (available for this config: {sorted(attrs)})")
        return attrs[name].copy()
