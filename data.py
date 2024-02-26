from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch
import pickle
import os
import numpy as np
from custom_sampler import CustomBatchSampler


class Dataset(InMemoryDataset):
    def __init__(self, root, data_directory, dataset_type, direction, use_node_information, word_dict, max_length, partial, transform=None, pre_transform=None):
        self.dataset_type = dataset_type
        self.direction = direction
        self.word_dict = word_dict
        self.sep = ' <SEP> '
        self.sep_list = []
        for char in self.sep:
            if char not in self.word_dict:
                self.word_dict[char] = len(self.word_dict)
            self.sep_list.append(self.word_dict[char])
        self.max_length = max_length
        self.use_node_information = use_node_information  # one of str, node, strnode, strnode_square, strnode_circle, str is deprecated
        self.data_directory = data_directory
        if partial == 1:
            self.intermediate_directory = '{0}_{1}_{2}'.format(max_length, direction, use_node_information)
        else:
            self.intermediate_directory = '{0}_{1}_{2}_{3}'.format(max_length, direction, use_node_information, partial)
        self.save_directory = data_directory + 'processed/' + self.intermediate_directory
        self.partial = partial
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.intermediate_directory + '/{0}_processed.dataset'.format(self.dataset_type)]

    def download(self):
        pass

    def process(self):
        with open('{0}{1}_dataset.pkl'.format(self.data_directory, self.dataset_type), 'rb') as f:
            dataset = pickle.load(f)
        data_list = []
        include_list = np.random.choice(len(dataset), int(self.partial*len(dataset)), replace=False)
        for i in range(len(dataset)):
            if i not in include_list:
                continue
            proof = dataset[i]
            node_features = []  # N x num_features
            labels = []  # (N, )
            # proof[1] list original source leaves to root
            # proof[2] list original target
            if self.use_node_information == 'node':
                source_nodes = []
                target_nodes = []
            else:
                if self.direction == 'lr':
                    source_nodes = proof[1]
                    target_nodes = proof[2]
                elif self.direction == 'rl':
                    source_nodes = proof[2]
                    target_nodes = proof[1]
                elif self.direction == 'bi':
                    source_nodes = proof[1] + proof[2]
                    target_nodes = proof[2] + proof[1]
                else:
                    raise NotImplementedError
            nodes = proof[3]
            for node in nodes:
                if self.use_node_information in ['strnode', 'node', 'str']:
                    all_feature = node[0] + self.sep_list + node[1]
                elif self.use_node_information == 'strnode_square':
                    all_feature = node[1]
                elif self.use_node_information == 'strnode_circle':
                    all_feature = node[0]
                else:
                    raise NotImplementedError
                labels.append(node[2])
                if self.use_node_information == 'str':
                    # overwrite feature
                    all_feature = [len(self.word_dict)]*10  # use a very small number here for number of tokens
                elif self.use_node_information in ['strnode', 'node', 'strnode_square', 'strnode_circle']:
                    if len(all_feature) > self.max_length:
                        all_feature = all_feature[:self.max_length]
                else:
                    raise NotImplementedError
                all_feature = torch.Tensor(all_feature)
                node_features.append(all_feature)
            y = torch.tensor(labels, dtype=torch.float)
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            data = Data(y=y, edge_index=edge_index)
            data.num_nodes = len(node_features)
            data.node_features = node_features
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_data(data_path, dataset_type, word_dict, max_length, batch_size, direction, use_node_information, num_workers, shuffle, partial, num_nodes_limit_per_batch):
    if dataset_type == 'valid':
        assert partial == 1
    dataset = Dataset(data_path, data_path, dataset_type, direction, use_node_information, word_dict, max_length, partial)
    custom_batch_sampler = CustomBatchSampler(dataset, shuffle, batch_size, num_nodes_limit_per_batch)
    loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=custom_batch_sampler)
    # loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader
