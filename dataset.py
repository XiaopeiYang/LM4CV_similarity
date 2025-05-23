from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imp_samp
import os
import json
import random
import yaml
import argparse
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

imp_samp_params = {
    "patch_size": 512,
    "reduce_factor": 1,
    "scale_dog": 1,
    "grid_sep": 256,
    "map_type": "importance",
    "patches_per_image": cfg['n_crops_per_image'],
    "blur_samp_map": False,
    "seed": 123
}

def important_crops_per_image(image, n_crops,imp_samp_params):
    crop_list=[]
    patcher = imp_samp.Patcher(image_path=image, **imp_samp_params)
    for i in range(n_crops):
        crop = next(patcher)
        crop_list.append(crop)    
    return crop_list


def important_random_size_crops_per_image(image, n_crops, imp_samp_params):
    crop_list = []
    for i in range(n_crops):
        # Randomly select crop size for each crop
        random_patch_size = random.randint(256, 512)
        # Update crop size parameters
        imp_samp_params['patch_size'] = random_patch_size
        imp_samp_params['grid_sep'] = int(random_patch_size/2)
        patcher = imp_samp.Patcher(image_path=image, **imp_samp_params) 
        crop = next(patcher)
        crop_list.append(crop)
    return crop_list

class FungiSmall(Dataset):
    """dataset for fungi"""

    def __init__(self, images_path: list, images_class: list, transform=None, use_patches=cfg['use_patches']):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.use_patches = use_patches  # This flag controls whether to use patches or resize images

        self.patches = []
        self.labels = []
        #self.patch_to_image_map = []  # Add a new list to store the original image index corresponding to each patch
        self._generate_patches()
        #print(self.labels)

    def _generate_patches(self):
        """Generates all important patches and their labels."""
        if self.use_patches:
            for img_path, img_label in zip(self.images_path, self.images_class):
                if cfg['IMP_SAMP_fix']:
                    patches = important_crops_per_image(img_path, cfg['n_crops_per_image'], imp_samp_params)
                else:
                    patches = important_random_size_crops_per_image(img_path, cfg['n_crops_per_image'], imp_samp_params)
                for patch in patches:
                    self.patches.append(patch)
                    self.labels.append(img_label)  # Assuming the same label for all patches from the same image
                    #self.patch_to_image_map.append(img_idx)  # Store the original image index corresponding to the patch
        else:
             # If not using patches, simply store the image paths and their labels
            self.patches = self.images_path
            self.labels = self.images_class
            #self.patch_to_image_map = list(range(len(self.images_path)))  # Each image corresponds to its own index

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, item):
        if self.use_patches:
            patch = self.patches[item]
            label = self.labels[item]
        #print("patch,label",patch,label)

            if self.transform:
                patch = self.transform(patch)
        else:
             # If not using patches, load the full image
            image_path = self.patches[item]
            label = self.labels[item]
            patch = Image.open(image_path)  # Load the full image
            
            if self.transform:
                patch = self.transform(patch)


        return patch, label

def remap_labels(labels, class_map):
    remapped_labels = [class_map[label] for label in labels]
    return remapped_labels

def read_split_data(json_file: str, root: str, base_ratio: float = cfg['base_ratio'], seed: int = cfg['base_ratio_seed'], few_shot: int = 10):
    # Check if paths exist
    assert os.path.exists(json_file), f"JSON file: {json_file} does not exist."
    assert os.path.exists(root), f"Root path: {root} does not exist."

    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Initialize path and tag lists
    train_images_paths = []
    train_images_labels = []
    test_images_paths = []
    test_images_labels = []


    # Process training data
    for item in data['train']:
        full_path = os.path.join(root, item[0])
        train_images_paths.append(full_path)
        train_images_labels.append(item[1])

    # Process test data
    for item in data['val']:
        full_path = os.path.join(root, item[0])
        test_images_paths.append(full_path)
        test_images_labels.append(item[1])
    
    # Split the classes into base and novel
    all_labels = list(set(train_images_labels))
    random.seed(seed)
    random.shuffle(all_labels)
    
    # num_base_classes = 3
    # base_classes =[0,1,5]
    # novel_classes = [2,3,4]
    
    num_base_classes = int(len(all_labels) * base_ratio)
    base_classes = all_labels[:num_base_classes]
    novel_classes = all_labels[num_base_classes:]
    
    # Sort base and novel classes to get the desired order
    base_classes.sort()
    novel_classes.sort()

    # Create a mapping from old labels to new labels
    class_map = {label: idx for idx, label in enumerate(base_classes)}
    class_map.update({label: idx + num_base_classes for idx, label in enumerate(novel_classes)})
    
    # Remap labels
    remapped_train_labels = remap_labels(train_images_labels, class_map)
    remapped_test_labels = remap_labels(test_images_labels, class_map)
    
    base_train_paths, base_train_labels = [], []
    base_test_paths, base_test_labels = [], []
    novel_test_paths, novel_test_labels = [], []
            
    if cfg['use_few_shot']: 
        # Collect few-shot samples for training
        base_class_samples = {label: [] for label in range(num_base_classes)}
    
        #print("base, novel",base_classes,novel_classes)
        for path, label in zip(train_images_paths, remapped_train_labels):
            if label < num_base_classes:
                base_class_samples[label].append((path, label))
                
        for label in base_class_samples:
            samples = base_class_samples[label]
            random.shuffle(samples)
            few_shot_samples = samples[:few_shot]
            for path, label in few_shot_samples:
                base_train_paths.append(path)
                base_train_labels.append(label)
    else:
        for path, label in zip(train_images_paths, remapped_train_labels):
            if label < num_base_classes:
                base_train_paths.append(path)
                base_train_labels.append(label)

    for path, label in zip(test_images_paths, remapped_test_labels):
        if label < num_base_classes:
            base_test_paths.append(path)
            base_test_labels.append(label)
        else:
            novel_test_paths.append(path)
            novel_test_labels.append(label)

    #print(f"{len(base_test_paths)} images for base testing.")
    #print(f"{len(novel_test_paths)} images for novel testing.")
    if base_ratio != 1.0:
        return (base_train_paths, base_train_labels), (base_test_paths, base_test_labels), (novel_test_paths, novel_test_labels),  (test_images_paths, test_images_labels),(train_images_paths,train_images_labels),base_classes, novel_classes
    else:
        return (base_train_paths, base_train_labels), (base_test_paths, base_test_labels)


class OnlineScoreDataset(Dataset):
    def __init__(self, attribute_embeddings, features, targets):
        self.features = torch.tensor(features).float()
        self.targets = torch.tensor(targets).float()
        self.attribute_embeddings = attribute_embeddings

    def __getitem__(self, idx):
        feature = self.features[idx]
        scores = feature.unsqueeze(0) @ self.attribute_embeddings.float().T
        scores = scores.squeeze()
        return scores, self.targets[idx]
        # return self.scores[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, group_array=None):
        self.features = torch.tensor(features)
        #print("features",features)
        #print("target:", targets)
        self.targets = torch.tensor(targets)
        self.group_array = group_array
        #print("self.features[idx]",self.features.size())
        #print("self.targets[idx]",self.targets.size()) 

    def __getitem__(self, idx):
        if self.group_array is not None:
            return self.features[idx], self.targets[idx], self.group_array[idx]
           
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)


