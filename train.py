import pickle
import pytorch_lightning as pl
from model_names import model_names_dict
from data import get_data
import argparse
import os
import json
import numpy as np
import git
import random
import torch
from utils import CustomCheckpoint


def main(args):
    data_path = args.data_path
    validation_path = args.validation_path
    max_length = args.max_length
    batch_size = args.batch_size
    num_epochs = args.epoch
    gpus = args.gpus
    save_path = args.save_directory
    checkpoint_directory = args.checkpoint_directory
    model = args.model
    direction = args.direction
    use_node_information = args.use_node_information
    num_workers = args.num_workers
    period = args.period
    partial_dataset = args.partial_dataset
    num_nodes_limit_per_batch = args.num_nodes_limit_per_batch
    shuffle = bool(args.shuffle)
    with open('{0}word_dict.pkl'.format(data_path), 'rb') as f:
        word_dict = pickle.load(f)
    train_loader = get_data(data_path, 'train', word_dict, max_length, batch_size, direction, use_node_information, num_workers=num_workers, shuffle=shuffle, partial=partial_dataset, num_nodes_limit_per_batch=num_nodes_limit_per_batch)
    valid_loader = get_data(validation_path, 'valid', word_dict, max_length, batch_size, direction, use_node_information, num_workers=num_workers, shuffle=False, partial=1, num_nodes_limit_per_batch=num_nodes_limit_per_batch)
    arg_dict['num_words'] = len(word_dict)
    model = model_names_dict[model](**arg_dict)
    all_checkpoints = [f for f in os.listdir(checkpoint_directory) if os.path.isfile(checkpoint_directory + f) and 'epoch' in f]
    if len(all_checkpoints) > 0:
        print(all_checkpoints)
        checkpoint_epochs = [int(e.replace('epoch=', '').replace('.ckpt', '')) for e in all_checkpoints]
        last_checkpoint_index = checkpoint_epochs.index(max(checkpoint_epochs))
        last_checkpoint = all_checkpoints[last_checkpoint_index]
        print('resuming checkpoint: {0}'.format(last_checkpoint))
        checkpoint_callback = CustomCheckpoint(checkpoint_directory=checkpoint_directory, save_directory=save_path, period=period)
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint_directory + last_checkpoint, max_epochs=num_epochs, gpus=gpus, default_root_dir=save_path, checkpoint_callback=checkpoint_callback, num_sanity_val_steps=0)
    else:
        checkpoint_callback = CustomCheckpoint(checkpoint_directory=checkpoint_directory, save_directory=save_path, period=period)
        trainer = pl.Trainer(max_epochs=num_epochs, gpus=gpus, default_root_dir=save_path, checkpoint_callback=checkpoint_callback, num_sanity_val_steps=0)
    trainer.fit(model, train_loader, valid_loader)
    checkpoint_callback.save_checkpoint_atomic(trainer, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train model")
    # I/O training config
    parser.add_argument('-d', dest='data_path', type=str, default='dataset/propositional_mm/')  # train dataset path
    parser.add_argument('-v', dest='validation_path', type=str, default='dataset/propositional_mm/')  # validation dataset path
    parser.add_argument('-save', dest='save_directory', type=str, default='results/')  # refers to log directory
    parser.add_argument('-checkpoint', dest='checkpoint_directory', type=str, default='checkpoints/')  # refers to checkpoint directory
    parser.add_argument('-name', dest='experiment_name', type=str, default='experiment0')  # experiment name
    parser.add_argument('-g', dest='gpus', type=int, default=1)
    parser.add_argument('-num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('-seed', dest='seed', type=int, default=47)
    parser.add_argument('-period', dest='period', type=int, default=1)
    # data
    parser.add_argument('-b', dest='batch_size', type=int, default=4)
    parser.add_argument('-direction', dest='direction', type=str, default='bi')  # rl: root to leaves, lr: leaves to root, bi: bidirectional
    parser.add_argument('-max_length', dest='max_length', type=int, default=512)
    parser.add_argument('-use_node_information', dest='use_node_information', type=str, default='strnode')
    parser.add_argument('-shuffle', dest='shuffle', type=int, default=1)
    parser.add_argument('-partial', dest='partial_dataset', type=float, default=1.0)  # for training set only
    parser.add_argument('-node_limit', dest='num_nodes_limit_per_batch', type=int, default=-1)
    # model related
    parser.add_argument('-l', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('-lam', dest='lam', type=float, default=1e-3)
    parser.add_argument('-ams', dest='ams_grad', type=int, default=0)
    parser.add_argument('-eps', dest='eps', type=float, default=1e-8)
    parser.add_argument('-average', dest='average', type=str, default='proof')
    parser.add_argument('-e', dest='epoch', type=int, default=500)
    parser.add_argument('-m', dest='model', type=str, default='sage_parallel')
    parser.add_argument('-embed_dim', dest='embed_dim', type=int, default=128)
    parser.add_argument('-e_agg', dest='embed_agg', type=str, default='sum')
    parser.add_argument('-pos', dest='pos_embed', type=int, default=0)
    parser.add_argument('-mlp_after_embed', dest='mlp_after_embed', type=int, default=1)
    parser.add_argument('-lbi', dest='lstm_bidirectional', type=int, default=1)
    parser.add_argument('-lho', dest='last_hidden_only', type=int, default=1)
    parser.add_argument('-num_layers', dest='num_layers', type=int, default=10)
    parser.add_argument('-lstm_num_layers', dest='lstm_num_layers', type=int, default=1)
    parser.add_argument('-lstm_dropout', dest='lstm_dropout', type=float, default=0.0)
    parser.add_argument('-hidden', dest='hidden', type=int, default=256)
    parser.add_argument('-scale', dest='scaling_factor', type=float, default=1)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    arg_dict = vars(args)
    arg_dict['git_hash'] = sha
    print(arg_dict)
    args.save_directory = args.save_directory + arg_dict['git_hash'] + '/' + args.experiment_name + '/'
    args.checkpoint_directory = args.checkpoint_directory + arg_dict['git_hash'] + '/' + args.experiment_name + '/'
    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)
    if not os.path.isdir(args.checkpoint_directory):
        os.makedirs(args.checkpoint_directory)
    with open(args.save_directory + 'args.json', 'w') as fp:
        json.dump(arg_dict, fp)
    main(args)
    print('done')
