import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class Network(nn.Module):
    def __init__(self, config_dim, hidden_dim=32):
        super(Network, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(config_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            # nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            # nn.ReLU(),
            # nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            # nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, config, similarities):
        # distances = similarity(meta_features)
        config = nn.BatchNorm1d(config.shape[1])(config)
        input_dense = torch.cat((config, similarities), 1)
        output_dense = self.dense(input_dense)
        return output_dense


# def similarity(meta_features):
#     with open("./data/metafeatures.json", "r") as f:
#         metafeatures = json.load(f)
#     used_meta = [k for k, v in metafeatures['higgs'].items() if not np.isnan(v)]
#     batch_similarities = []
#     for metafeature in meta_features:
#         similarities = []
#         for dataset in 'adult', 'higgs', 'vehicle', 'volkert':
#             meta = {k: v for k, v in metafeatures[dataset].items() if k in used_meta}
#             train_meta_vector = np.array(list(meta.values())).reshape(1, len(used_meta))
#             similarities.append(1 - cosine_distances(train_meta_vector, meta_features).squeeze().item(0))
#         batch_similarities.append(similarities)
#     return torch.Tensor(batch_similarities)
