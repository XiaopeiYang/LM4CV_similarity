import os
import sys
import pdb
import yaml
from utils.train_utils import *
from cluster import cluster
import torch
import torch.nn.functional as F

with_attributes = True

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

def evaluate_novel_classes(attribute_embeddings, novel_test_loader):
    image_embeddings = []
    image_labels = []

    # Extract image embeddings and tags from the loader
    for embeddings, labels in novel_test_loader:
        #print(embeddings.size())
        #print(labels)
        image_embeddings.append(embeddings)
        image_labels.append(labels)

    #print("image_embeddings",image_embeddings)
    #print("image_labels",image_labels)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_labels = torch.cat(image_labels, dim=0)
    #print("image_embeddings",image_embeddings.size())
    image_embeddings = image_embeddings.float()

    # Ensure image_embeddings are integer type for indexing
    #print(f"Sample embeddings: {image_embeddings}")
    #print(f"Sample labels: {image_labels}")

    # 将 image_embeddings 映射到 attribute_embeddings 空间
    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    novel_embeddings = torch.matmul(image_embeddings, attribute_embeddings_tensor.t())
    #print("attribute_embeddings_tensor",attribute_embeddings_tensor.size())
    #print("novel_embeddings",novel_embeddings.size())

    # 计算余弦相似度矩阵
    if with_attributes:
        similarity_matrix = calculate_cosine_similarity(novel_embeddings)
    else:
        similarity_matrix = calculate_cosine_similarity(image_embeddings)

    # 图像检索，获取前 1 个最相似的图像（可以根据需要调整 top_k）
    retrieval_results = image_retrieval(similarity_matrix, top_k=1)

    # 计算准确率
    accuracy = calculate_accuracy(retrieval_results, image_labels)
    #print(f"Distribution of labels in test set: {torch.Size(image_labels)}")

    # Check some of the embeddings and labels to verify data integrity
    

    return accuracy

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
    _,novel_test_loader,_ =get_feature_dataloader(cfg)
    accuracy = evaluate_novel_classes(attributes_embeddings, novel_test_loader)
    print(f"Accuracy: {accuracy}")
    # Additional suggestions to troubleshoot accuracy issues
    print(f"Total number of test samples: {len(novel_test_loader.dataset)}")
    

    return _


if __name__ == '__main__':
    
    main(cfg)


