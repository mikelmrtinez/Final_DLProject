import pandas as pd
import numpy as np
import torch


def droped_conf_features(X) -> [str]:
    train_conf_df = pd.DataFrame(data=list(X))
    droped_columns = []
    for column in train_conf_df.columns:
        if (len(train_conf_df[column].unique()) == 1):
            droped_columns.append(column)
    return droped_columns


def kept_meta_features(metafeatures, name='higgs') -> np.array:
    return [k for k, v in metafeatures['higgs'].items() if not np.isnan(v)]


def config_list_to_tensor(X: np.array([dict]), droped_configs: [str]) -> torch.Tensor:
    conf_df = pd.DataFrame(data=list(X))
    conf_df.drop(droped_configs, axis=1, inplace=True)
    return torch.Tensor(conf_df.to_numpy(dtype="Float16"))


def metadata_list_to_tensor(metafeatures_dict: dict, dataset: [str], kept_meta: [str]) -> torch.Tensor:
    metafeatures = []
    for name in dataset:
        meta_raw = metafeatures_dict[str(name)]
        meta = {k: v for k, v in meta_raw.items() if k in kept_meta}
#         print(list(meta.values()))
        metafeatures.append(torch.Tensor(list(meta.values())).reshape(1, len(kept_meta)))
    return torch.cat(metafeatures, 0)
