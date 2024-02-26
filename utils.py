import os
import pytorch_lightning as pl
import pickle
from collections import Counter
import numpy as np
import copy


class CustomCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, checkpoint_directory, save_directory, period):
        # checkpoints in checkpoint directory are 0-base index, in save directory are 1-base
        super().__init__(filepath=checkpoint_directory, period=period, save_top_k=None, monitor=None)
        self.checkpoint_directory = checkpoint_directory
        self.period = period
        self.save_directory = save_directory  # used for last save

    def save_checkpoint_atomic(self, trainer, save_path):
        temp_path = os.path.join(os.path.dirname(save_path), "temp.ckpt")
        trainer.save_checkpoint(temp_path)
        os.replace(temp_path, save_path + 'epoch={0}.ckpt'.format(trainer.current_epoch))
        if os.path.isfile(save_path + 'epoch={0}.ckpt'.format(trainer.current_epoch - self.period)):
            os.remove(save_path + 'epoch={0}.ckpt'.format(trainer.current_epoch - self.period))

    def on_validation_end(self, trainer, pl_module):
        """
        checkpoints can be saved at the end of the val loop
        """
        if (trainer.current_epoch + 1) % self.period == 0:
            print('saving checkpoint at epoch {0}'.format(trainer.current_epoch))
            self.save_checkpoint_atomic(trainer, self.checkpoint_directory)


def subtract_datasets(d_1, d_2):
    # d_1 - d_2
    names_2 = [e[0] for e in d_2]
    res = []
    for i in range(len(d_1)):
        e = d_1[i]
        if e[0] not in names_2:
            res.append(e)
    return res


def union_datasets(d_1, d_2):
    names_1 = [e[0] for e in d_1]
    names_2 = [e[0] for e in d_2]
    if len(names_1) > len(names_2):
        res = copy.deepcopy(d_1)
        for i in range(len(d_2)):
            e = d_2[i]
            if e[0] not in names_1:
                res.append(e)
    else:
        res = copy.deepcopy(d_2)
        for i in range(len(d_1)):
            e = d_1[i]
            if e[0] not in names_2:
                res.append(e)
    return res


def merge_datasets(dataset_path_1, dataset_path_2):
    with open(dataset_path_1 + 'train_dataset.pkl', 'rb') as f:
        train_dataset_1 = pickle.load(f)
    with open(dataset_path_1 + 'valid_dataset.pkl', 'rb') as f:
        valid_dataset_1 = pickle.load(f)
    with open(dataset_path_2 + 'train_dataset.pkl', 'rb') as f:
        train_dataset_2 = pickle.load(f)
    with open(dataset_path_2 + 'valid_dataset.pkl', 'rb') as f:
        valid_dataset_2 = pickle.load(f)
    difference_1 = subtract_datasets(valid_dataset_1, train_dataset_2)
    difference_2 = subtract_datasets(valid_dataset_2, train_dataset_1)
    res = union_datasets(difference_1, difference_2)
    return res


def merge_datasets_2(dataset_path_1, dataset_path_2, output_path):
    with open(dataset_path_1 + 'valid_dataset.pkl', 'rb') as f:
        valid_dataset_1 = pickle.load(f)
    with open(dataset_path_1 + 'test_dataset.pkl', 'rb') as f:
        test_dataset_1 = pickle.load(f)
    with open(dataset_path_2 + 'train_dataset.pkl', 'rb') as f:
        train_dataset_2 = pickle.load(f)
    difference_1 = subtract_datasets(valid_dataset_1, train_dataset_2)
    difference_2 = subtract_datasets(test_dataset_1, train_dataset_2)
    res = union_datasets(difference_1, difference_2)

    theorem_names = []
    for e in res:
        name = e[0]
        expanding_theorem = name[name.find('expand_') + 7:name.find('_in_')]
        theorem_names.append(expanding_theorem)
    counts = list(Counter(theorem_names).values())
    with open(output_path + 'valid_dataset.pkl', 'wb') as f:
        pickle.dump(res, f)
    np.save(output_path + 'expanding_theorem_histogram_valid.npy', counts)


def filter_nodes_to_tokens(load_path, output_path, token_length):
    with open(load_path + 'train.src', 'r') as f:
        train_source = f.readlines()
    with open(load_path + 'train.tgt', 'r') as f:
        train_target = f.readlines()
    with open(load_path + 'train_proof_names.pkl', 'rb') as f:
        train_proof_names = pickle.load(f)
    assert len(train_source) == len(train_target)
    assert len(train_target) == len(train_proof_names)
    delete_indices = []
    for i in range(len(train_source)):
        source = train_source[i]
        if len(source.split()) > token_length:
            delete_indices.append(i)

    for i in range(len(train_target)):
        target = train_target[i]
        if len(target.split()) > token_length:
            delete_indices.append(i)
    delete_indices = list(set(delete_indices))
    for i in range(len(train_source) - 1, -1, -1):
        if i in delete_indices:
            del train_source[i]
            del train_target[i]
            del train_proof_names[i]
    with open(output_path + 'train.src', 'w') as f:
        f.writelines(train_source)
    with open(output_path + 'train.tgt', 'w') as f:
        f.writelines(train_target)
    with open(output_path + 'train_proof_names.pkl', 'wb') as f:
        pickle.dump(train_proof_names, f)
