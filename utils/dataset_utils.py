import os
import sys
import pdb
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
import torch
import torch.nn as nn
import pandas as pd
import torchvision
import transformers
from collections import defaultdict
import sklearn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from dataset import read_split_data, FungiSmall
import yaml

import clip


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default='./configs/fungi.yaml',
                        help='configurations for training')
    return parser.parse_args()
args = parse_config()
with open(f'{args.config}', "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

json_file = cfg['json_file']
ROOT = cfg['ROOT']
use_patches = cfg['use_patches']
n_crops_per_image = cfg['n_crops_per_image']
#use_patches = False
#n_crops_per_image = 7

def clean_label(true_labels):
    true_labels = np.array(true_labels)
    if np.min(true_labels) > 0:
        true_labels -= np.min(true_labels)
    return true_labels

def get_labels(dataset):
        
    # Read and split data into base and novel classes
    (_, base_train_labels), (_, base_test_labels), (_, novel_test_labels),(_,test_images_labels) ,_, _ = read_split_data(json_file, ROOT)
       

    if use_patches:
    # If using patches, generate multiple labels for each image
        expanded_base_train_labels = []
        expanded_base_test_labels = []
        expanded_novel_test_labels = []
        expanded_test_labels = []

        for label in base_train_labels:
            expanded_base_train_labels.extend([label] * n_crops_per_image)

        for label in base_test_labels:
            expanded_base_test_labels.extend([label] * n_crops_per_image)

        for label in novel_test_labels:
            expanded_novel_test_labels.extend([label] * n_crops_per_image)
        for label in test_images_labels:
            expanded_test_labels.extend([label] * n_crops_per_image)
        base_train_labels = expanded_base_train_labels
        base_test_labels = expanded_base_test_labels
        novel_test_labels = expanded_novel_test_labels
        test_images_labels = expanded_test_labels   

    return base_train_labels,base_test_labels,novel_test_labels,test_images_labels

def get_image_dataloader(dataset, preprocess, preprocess_eval=None, shuffle=False):
    # Read and split data into base and novel classes
    (base_train_paths, base_train_labels), (base_test_paths, base_test_labels), (novel_test_paths, novel_test_labels),  (test_images_paths, test_images_labels),base_classes, novel_classes = read_split_data(json_file, ROOT)
        
    print(f"{len(base_train_paths)} images for base training.")
    print(f"{len(base_test_paths)} images for base testing.")
    print(f"{len(novel_test_paths)} images for novel testing.")
    print(f"{len(test_images_labels)} images for all testing.")
    print("base_classes", base_classes)
    print("novel_classes", novel_classes)
    ########################################
    #print("novel_test_paths", novel_test_paths)
    #print("novel_test_labels", novel_test_labels)
    ########################################
    # Create datasets
    base_trainset = FungiSmall(images_path=base_train_paths, images_class=base_train_labels, transform=preprocess)
    base_testset = FungiSmall(images_path=base_test_paths, images_class=base_test_labels, transform=preprocess)
    novel_testset = FungiSmall(images_path=novel_test_paths, images_class=novel_test_labels, transform=preprocess)
    all_testset = FungiSmall(images_path=test_images_paths, images_class=test_images_labels, transform=preprocess)

    # Create data loaders
    base_train_loader = DataLoader(base_trainset, batch_size=4, shuffle=False)
    base_test_loader = DataLoader(base_testset, batch_size=4, shuffle=False)
    novel_test_loader = DataLoader(novel_testset, batch_size=4, shuffle=False)
    all_test_loader = DataLoader(all_testset, batch_size=4, shuffle=False)
 

    return  base_train_loader,base_test_loader,novel_test_loader,all_test_loader

def get_output_dim(dataset):
    #print(len(np.unique(get_labels(dataset)[0])))
    return len(np.unique(get_labels(dataset)[0]))

def get_attributes(cfg):
    if cfg['attributes'] == 'random':
        '''
        Generate random attributes
        '''
        import urllib.request
        import random

        word_url = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = urllib.request.urlopen(word_url)
        long_txt = response.read().decode()
        word_list = long_txt.splitlines()

        print(len(word_list))

        random_words = []
        for i in range(512):
            words = random.choices(word_list, k=random.randint(1, 5))
            random_words.append(' '.join(words))

        attributes = random_words
        return attributes
    elif cfg['attributes'] == 'fungi':
        return open("./data/fungi/fungi_attributes.txt", 'r').read().strip().split("\n")

def get_prefix(cfg):
    if cfg['dataset'] == 'fungi':
        return "A photo of a fungi with "