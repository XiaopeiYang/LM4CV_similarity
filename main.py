import os
import sys
import pdb
import yaml
from utils.train_utils import *
from cluster import cluster
import torch
import torch.nn.functional as F

#with_attributes = True
#evaluate_with_novel_classes = True

def calculate_cosine_similarity(embeddings):

    # Calculate cosine similarity matrix
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    print("norm_embedding",norm_embeddings.size())
    similarity_matrix = torch.einsum('ik,jk->ij', norm_embeddings, norm_embeddings)
    print("similarity_matrix",similarity_matrix.size())
    return similarity_matrix

def image_retrieval(similarity_matrix, top_k=1):
    # Image retrieval based on similarity to find the most similar image
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k + 1, dim=1, largest=True, sorted=True)
    retrieval_results = top_k_indices[:, 1:]  # exclude oneself
    print("retrieval_results",retrieval_results.size())
    return retrieval_results

def calculate_accuracy(retrieval_results, labels):
    '''
    Calculate the accuracy of similarity image(with attributes) results'''
    # Calculate the accuracy of search results
    correct = 0
    total = 0
    
    for idx, retrieved_indices in enumerate(retrieval_results):
        #print("retrieved_indices",retrieved_indices)
        #print("labels[retrieved_indices]",labels[retrieved_indices])
        #print("labels[idx]",labels[idx])
        if labels[idx] in labels[retrieved_indices]:
            #print(correct)
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

def evaluate_novel_classes(attribute_embeddings, test_loader):
    '''
    combine image embeddings with attribute embeddings to calculate the similarity matrix and evaluate the accuracy
    '''
    image_embeddings = []
    image_labels = []

    # Extract image embeddings and tags from the loader
    for embeddings, labels in test_loader:
        image_embeddings.append(embeddings)
        image_labels.append(labels)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_labels = torch.cat(image_labels, dim=0)
    #print("image_embeddings",image_embeddings.size())
    image_embeddings = image_embeddings.float()

    # Ensure image_embeddings are integer type for indexing
    #print(f"Sample embeddings: {image_embeddings}")
    #print(f"Sample labels: {image_labels}")

    # Map image_embeddings to attribute_embeddings space
    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    imagewithattributes_embeddings = torch.matmul(image_embeddings, attribute_embeddings_tensor.t())
    #print("attribute_embeddings_tensor",attribute_embeddings_tensor.size())
    #print("novel_embeddings",novel_embeddings.size())

    # Calculate cosine similarity matrix
    if cfg['with_attributes']:
        similarity_matrix = calculate_cosine_similarity(imagewithattributes_embeddings)
    else:
        similarity_matrix = calculate_cosine_similarity(image_embeddings)

    # Image retrieval, get the top 1 most similar image (top_k can be adjusted as needed)
    retrieval_results = image_retrieval(similarity_matrix, top_k=1)

    # Calculate accuracy
    accuracy = calculate_accuracy(retrieval_results, image_labels)
    #print(f"Distribution of labels in test set: {torch.Size(image_labels)}")

    # Check some of the embeddings and labels to verify data integrity
    

    return accuracy

def remap_keys_in_dict(input_dict, new_keys):
    """
    Replace the keys in the input dictionary with the new keys provided.

    Parameters:
    input_dict (dict): The original dictionary with keys to be replaced.
    new_keys (list): A list of new keys to replace the old keys.

    Returns:
    dict: A new dictionary with the replaced keys.
    """
    if len(new_keys) != len(input_dict):
        raise ValueError("The number of new keys must match the number of keys in the input dictionary.")
    
    new_dict = {}
    for new_key, old_key in zip(new_keys, list(input_dict.keys())):
        new_dict[new_key] = input_dict[old_key]
    
    return new_dict

def attribute_with_high_score(attribute_embeddings, test_loader,attributes):
    '''
    for each class of fungi, choose attributes with highest importance scores
    '''
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

    # Group embeddings by labels
    label_to_embeddings = defaultdict(list)
    for embedding, label in zip(image_embeddings, image_labels):
        label_to_embeddings[label.item()].append(embedding)


    # Calculate the average embedding for each label
    label_to_avg_embedding = {}
    for label, embeddings in label_to_embeddings.items():
        embeddings_tensor = torch.stack(embeddings)
        avg_embedding = torch.mean(embeddings_tensor, dim=0)
        label_to_avg_embedding[label] = avg_embedding
        print("label_to_avg_embedding",label_to_avg_embedding[label].size())

    # Calculate imagewithattributes_embeddings
    label_to_imagewithattributes = {}
    for label, avg_embedding in label_to_avg_embedding.items():
        imagewithattributes_embedding = torch.matmul(avg_embedding, attribute_embeddings_tensor.t())
        label_to_imagewithattributes[label] = imagewithattributes_embedding
        
        print("label_to_imagewithattributes",label_to_imagewithattributes[label].size())

    # Find the top values and their corresponding attributes for each label
    top_k = 5  # You can set this to however many top attributes you want
    label_to_top_attributes = {}
    for label, imagewithattributes_embedding in label_to_imagewithattributes.items():
        top_values, top_indices = torch.topk(imagewithattributes_embedding, top_k)
        print("top_indices",top_indices)
        top_indices_list = top_indices.tolist()
        top_attributes = [attributes[i] for i in top_indices_list]
        label_to_top_attributes[label] = (top_values, top_attributes)
    
    if cfg['evaluate_with_novel_classes']:
        (_, _), (_, _), (_, _),  (_, _),_, novel_classes = read_split_data(json_file, ROOT)
        label_to_top_attributes = remap_keys_in_dict(label_to_top_attributes, novel_classes)

    return label_to_top_attributes

def main(cfg):

    set_seed(cfg['seed'])
    print(cfg)

    if cfg['cluster_feature_method'] == 'linear' and cfg['num_attributes'] != 'full':
        acc, model, attributes, attributes_embeddings = cluster(cfg)
        #print(attributes_embeddings)
    else:
        attributes, attributes_embeddings = cluster(cfg)

    if cfg['reinit']  and cfg['num_attributes'] != 'full':
        print("cfg['reinit']  and cfg['num_attributes'] != 'full'")
        assert cfg['cluster_feature_method'] == 'linear'
        feature_train_loader, feature_test_loader = get_feature_dataloader(cfg)
        #feature_train_loader, feature_test_loader = get_score_dataloader(cfg,attributes_embeddings)
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(dim=-1, keepdim=True)
        for param in model[0].parameters():
            param.requires_grad = False
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, feature_train_loader, feature_test_loader)
    '''
    else:
        model = get_model(cfg, cfg['score_model'], input_dim=len(attributes), output_dim=get_output_dim(cfg['dataset']))
        score_train_loader, score_test_loader = get_score_dataloader(cfg, attributes_embeddings)
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, score_train_loader, score_test_loader)
    '''
    if cfg['evaluate_with_novel_classes']:
        _,_,novel_test_loader,_ =get_feature_dataloader(cfg)
    
        accuracy = evaluate_novel_classes(attributes_embeddings, novel_test_loader)
        print(f"Accuracy: {accuracy}")
        # Additional suggestions to troubleshoot accuracy issues
        print(f"Total number of test samples: {len(novel_test_loader.dataset)}")
        attribute=attribute_with_high_score(attributes_embeddings, novel_test_loader,attributes)
        print(attribute)
    else:
        _,_,_,all_test_loader = get_feature_dataloader(cfg)
        accuracy = evaluate_novel_classes(attributes_embeddings, all_test_loader)
        print(f"Accuracy: {accuracy}")
        # Additional suggestions to troubleshoot accuracy issues
        print(f"Total number of test samples: {len(all_test_loader.dataset)}")
        attribute=attribute_with_high_score(attributes_embeddings, all_test_loader,attributes)
        print(attribute)
    

    return _


if __name__ == '__main__':
    
    main(cfg)


