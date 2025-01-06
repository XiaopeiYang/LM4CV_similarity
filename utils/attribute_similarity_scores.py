import torch
import torch.nn.functional as F
from .train_utils import read_split_data
from .rat1 import label_mapping, remap_keys_in_dict
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from collections import defaultdict

def attribute_with_all_images_score(cfg, attribute_embeddings, test_loader,attributes):
                        
    #for each class of fungi, choose attributes with highest importance scores
    #Select all images from each class,and calculate the similarity score with each attribute
    
    image_embeddings = []
    image_labels = []

    # Extract image embeddings and tags from the loader
    for embeddings, labels in test_loader:
        image_embeddings.append(embeddings)
        image_labels.append(labels)
        
    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_labels = torch.cat(image_labels, dim=0)
    image_embeddings = image_embeddings.float()

    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    torch.save(attribute_embeddings_tensor, 'attribute_embeddings_tensor.pth')

    
    # Group embeddings by labels
    label_to_embeddings = defaultdict(list)
    for embedding, label in zip(image_embeddings, image_labels):
        label_to_embeddings[label.item()].append(embedding)
        
        
    # Calculate image-with-attributes embeddings first, then average for each label
    label_to_imagewithattributes = defaultdict(list)
    for label, embeddings in label_to_embeddings.items():
        embeddings_tensor = torch.stack(embeddings)
        norm_embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
        imagewithattributes_embedding = torch.matmul(norm_embeddings_tensor, attribute_embeddings_tensor.t())
        label_to_imagewithattributes[label].append(imagewithattributes_embedding)
    
    
    # Calculate the average image-with-attributes embedding for each label    
    label_to_avg_imagewithattributes = {}  
    for label, imagewithattributes_embeddings in label_to_imagewithattributes.items():
        imagewithattributes_tensor = torch.cat(imagewithattributes_embeddings, dim=0)
        avg_imagewithattributes_embedding = torch.mean(imagewithattributes_tensor, dim=0)
        label_to_avg_imagewithattributes[label] = avg_imagewithattributes_embedding

        
    for label, avg_imagewithattributes_embedding in label_to_avg_imagewithattributes.items():
        data = avg_imagewithattributes_embedding.detach().cpu().numpy()
        #Custom nonlinear transformations: Combining logarithmic and exponential transformations
        custom_transformed_data = np.exp(np.log(data) * 3)
        plt.figure(figsize=(15, 5))    # Increase figure size to accommodate labels
        plt.bar(range(len(custom_transformed_data)), custom_transformed_data)
        label_name = label_mapping.get(label, f"Label {label}")
        plt.title(f"Attributes similarity scores for {label_name}",fontsize=20)
        plt.xlabel('Index of Attributes',fontsize=15)  # Reflects the total number of elements
        plt.ylabel('scores',fontsize=15)  # Compares each data's size
        plt.yticks([])  # Remove the tick labels from the y-axis
        plt.xticks(range(len(attributes)), range(len(attributes)), rotation='vertical')  # Use attribute names for x-ticks
        plt.tight_layout()  # Adjust layout to make room for x-labels
        file_path = f'./histogram/all/{label_name}attributes.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        if os.path.exists(file_path):
           print(f"File saved successfully: {file_path}")
        else:
           print(f"Failed to save file: {file_path}")
           
#     for label, avg_imagewithattributes_embedding in label_to_avg_imagewithattributes.items():
#         data = avg_imagewithattributes_embedding.detach().cpu().numpy()
    
#         #print("data",data)
#         #Custom nonlinear transformations: Combining logarithmic and exponential transformations
#         plt.figure(figsize=(15, 5))    # Increase figure size to accommodate labels
#         plt.bar(range(len(data)), data)
#         label_name = label_mapping.get(label, f"Label {label}")
#         plt.title(f"Attributes similarity scores for {label_name}",fontsize=20)
#         plt.xlabel('Index of Attributes',fontsize=15)  # Reflects the total number of elements
#         plt.ylabel('scores',fontsize=15)  # Compares each data's size
#         plt.yticks([])  # Remove the tick labels from the y-axis
#         plt.xticks(range(len(attributes)), range(len(attributes)), rotation='vertical')  # Use attribute names for x-ticks
#         plt.tight_layout()  # Adjust layout to make room for x-labels
#         plt.savefig(f'{label_name}attributes.png', dpi=300, bbox_inches='tight')
#         plt.close()
        

    # # Find the top values and their corresponding attributes for each label
    # top_k = 1  # You can set this to however many top attributes you want
    # label_to_top_attributes = {}
    # for label, avg_imagewithattributes_embedding in label_to_avg_imagewithattributes.items():
    #     top_values, top_indices = torch.topk(avg_imagewithattributes_embedding, top_k)
    #     print("top_indices",top_indices)
    #     top_indices_list = top_indices.tolist()
    #     top_attributes = [attributes[i] for i in top_indices_list]
    #     label_to_top_attributes[label] = (top_values, top_attributes)
    
    # if cfg['evaluate_with_novel_classes']:
    #     (_, _), (_, _), (_, _),  (_, _),(_,_),_, novel_classes = read_split_data(cfg['json_file'], cfg['ROOT'])
    #     label_to_top_attributes = remap_keys_in_dict(label_to_top_attributes, novel_classes)

    return None

def attribute_with_random_image_score(cfg, attribute_embeddings, test_loader,attributes):
                        
    #for each class of fungi, choose attributes with highest importance scores
    #Randomly select one image from each class,and calculate the similarity score with each attribute

    # Initialize a list to store a random selection of every 100
    selected_embeddings = []
    selected_labels = []

    # Initialize counter and temporary list
    batch_embeddings = []
    batch_labels = []
    count = 0

    # Reading data from test_loader
    for embeddings, labels in test_loader:
        batch_embeddings.extend(embeddings)
        batch_labels.extend(labels)
        count += len(labels)
        
        # A random selection is performed every 100 data
        if count >= cfg['testing_samples_per_class']:
            while count >= cfg['testing_samples_per_class']:
                count -= cfg['testing_samples_per_class']
                
                # Randomly select one from the first 100 data in the current batch
                sub_embeddings = batch_embeddings[:cfg['testing_samples_per_class']]
                sub_labels = batch_labels[:cfg['testing_samples_per_class']]
                
                random_index = random.randint(0, cfg['testing_samples_per_class']-1)
                selected_embeddings.append(sub_embeddings[random_index])
                selected_labels.append(sub_labels[random_index])
                
                # Remove the first 100 processed data
                batch_embeddings = batch_embeddings[cfg['testing_samples_per_class']:]
                batch_labels = batch_labels[cfg['testing_samples_per_class']:]

    # Processing data with less than 100 items remaining
    if batch_labels:
        random_index = random.randint(0, len(batch_labels) - 1)
        selected_embeddings.append(batch_embeddings[random_index])
        selected_labels.append(batch_labels[random_index])

    # print the results of the random selection
    # for embedding,label in zip(selected_embeddings,selected_labels):
    #     print("choose random label:", label)
    #     print("choose random embedding:", embedding)


    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    #torch.save(attribute_embeddings_tensor, 'attribute_embeddings_tensor.pth')

    
    # Group embeddings by labels
    label_to_embeddings = defaultdict(list)
    for embedding, label in zip(selected_embeddings,selected_labels):
        label_to_embeddings[label.item()].append(embedding)
        
        
    # Calculate image-with-attributes embeddings first, then average for each label
    label_to_imagewithattributes = {}
    for label, embeddings in label_to_embeddings.items():
        embeddings_tensor = torch.stack(embeddings)
        embeddings_tensor = embeddings_tensor.float()
        imagewithattributes_embedding = torch.matmul(embeddings_tensor, attribute_embeddings_tensor.t())
        imagewithattributes_embedding = torch.squeeze(imagewithattributes_embedding, dim=0)
        label_to_imagewithattributes[label]=imagewithattributes_embedding
        
    for label, imagewithattributes_embedding in label_to_imagewithattributes.items():    
        data = imagewithattributes_embedding.detach().cpu().numpy()
        #Custom nonlinear transformations: Combining logarithmic and exponential transformations
        custom_transformed_data = np.exp(np.log(data) * 3)
        plt.figure(figsize=(15, 5))    # Increase figure size to accommodate labels
        plt.bar(range(len(custom_transformed_data)), custom_transformed_data)
        label_name = label_mapping.get(label, f"Label {label}")
        plt.title(f"Attributes similarity scores for {label_name}",fontsize=20)
        plt.xlabel('Index of Attributes',fontsize=15)  # Reflects the total number of elements
        plt.ylabel('scores',fontsize=15)  # Compares each data's size
        plt.yticks([])  # Remove the tick labels from the y-axis
        plt.xticks(range(len(attributes)), range(len(attributes)), rotation='vertical')  # Use attribute names for x-ticks
        plt.tight_layout()  # Adjust layout to make room for x-labels
        file_path = f'./histogram/random/{label_name}attributes.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        if os.path.exists(file_path):
           print(f"File saved successfully: {file_path}")
        else:
           print(f"Failed to save file: {file_path}")
    
    # for label, imagewithattributes_embedding in label_to_imagewithattributes.items():
    #     data = imagewithattributes_embedding.detach().cpu().numpy()
    #     #print("data",data)
    #     #Custom nonlinear transformations: Combining logarithmic and exponential transformations
    #     plt.figure(figsize=(15, 5))    # Increase figure size to accommodate labels
    #     plt.bar(range(len(data)), data)
    #     label_name = label_mapping.get(label, f"Label {label}")
    #     plt.title(f"Attributes similarity scores for {label_name}",fontsize=20)
    #     plt.xlabel('Index of Attributes',fontsize=15)  # Reflects the total number of elements
    #     plt.ylabel('scores',fontsize=15)  # Compares each data's size
    #     plt.yticks([])  # Remove the tick labels from the y-axis
    #     plt.xticks(range(len(attributes)), range(len(attributes)), rotation='vertical')  # Use attribute names for x-ticks
    #     plt.tight_layout()  # Adjust layout to make room for x-labels
    #     histogram_dir = './histogram'
    #     if not os.path.exists(histogram_dir):
    #         os.makedirs(histogram_dir)
    #     plt.savefig(f'./histogram/{label_name}attributes.png', dpi=300, bbox_inches='tight')
    #     plt.close()
        

    # # Find the top values and their corresponding attributes for each label
    # top_k = 1  # You can set this to however many top attributes you want
    # label_to_top_attributes = {}
    # for label, imagewithattributes_embedding in label_to_imagewithattributes.items():
    #     top_values, top_indices = torch.topk(imagewithattributes_embedding, top_k)
    #     print("top_indices",top_indices)
    #     top_indices_list = top_indices.tolist()
    #     top_attributes = [attributes[i] for i in top_indices_list]
    #     label_to_top_attributes[label] = (top_values, top_attributes)
    
    # if cfg['evaluate_with_novel_classes']:
    #     (_, _), (_, _), (_, _),  (_, _),(_,_),_, novel_classes = read_split_data(cfg['json_file'], cfg['ROOT'])
    #     label_to_top_attributes = remap_keys_in_dict(label_to_top_attributes, novel_classes)

    return None