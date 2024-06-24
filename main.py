import os
import sys
import pdb
import yaml
from utils.train_utils import *
from cluster import cluster
import torch
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

label_mapping = {
    0: "Caspofungin",
    1: "Carbendazim(unknown)",
    2: "Mycelium(unknown)",
    3: "Germination Inhibition(unknown)",
    4: "GWT1(unknown)",
    5: "Tebuconazole"
}

without_zscore=True
with_zscore=True
#check whether normliazed  
def is_normalized(tensor, p=2, dim=1, atol=1e-5):
    norms = torch.norm(tensor, p=p, dim=dim)
    return torch.allclose(norms, torch.ones_like(norms), atol=atol)

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

def calculate_eachclass_accuracy(retrieval_results, labels):
    # Calculate the accuracy for each class
    class_correct = {}
    class_total = {}
    
    for idx, retrieved_indices in enumerate(retrieval_results):
        label = labels[idx].item()
        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0
        
        if label in labels[retrieved_indices]:
            class_correct[label] += 1
        class_total[label] += 1
    
    class_accuracy = {label: class_correct[label] / class_total[label] for label in class_correct}
    return class_accuracy

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
    eachclass_accuracy = calculate_eachclass_accuracy(retrieval_results, image_labels)
    # Check some of the embeddings and labels to verify data integrity
    

    return accuracy, eachclass_accuracy

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
        
        
    # Calculate image-with-attributes embeddings first, then average for each label
    label_to_imagewithattributes = defaultdict(list)
    for label, embeddings in label_to_embeddings.items():
        embeddings_tensor = torch.stack(embeddings)
        print("embeddings_tensor",embeddings_tensor.size())
        #print("Attribute embeddings normalized:", is_normalized(attribute_embeddings_tensor),torch.norm(attribute_embeddings_tensor, p=2, dim=1))
        #print("Image embeddings normalized:", is_normalized(embeddings_tensor),torch.norm(embeddings_tensor, p=2, dim=1))
        norm_embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
        #print("Nrom Image embeddings normalized:", is_normalized(norm_embeddings_tensor))
        print("norm_embeddings_tensor",norm_embeddings_tensor.size())
        imagewithattributes_embedding = torch.matmul(norm_embeddings_tensor, attribute_embeddings_tensor.t())
        #print("the norm of imagewithattributes_embedding:",torch.norm(imagewithattributes_embedding, p=2, dim=1))
        label_to_imagewithattributes[label].append(imagewithattributes_embedding)
    
    # Calculate the average image-with-attributes embedding for each label
    label_to_avg_imagewithattributes = {}
    for label, imagewithattributes_embeddings in label_to_imagewithattributes.items():
        imagewithattributes_tensor = torch.cat(imagewithattributes_embeddings, dim=0)
        avg_imagewithattributes_embedding = torch.mean(imagewithattributes_tensor, dim=0)
        label_to_avg_imagewithattributes[label] = avg_imagewithattributes_embedding
        print("label_to_avg_imagewithattributes", label_to_avg_imagewithattributes[label].size())

    ## Calculate the average embedding for each label
    #label_to_avg_embedding = {}
    #for label, embeddings in label_to_embeddings.items():
    #    embeddings_tensor = torch.stack(embeddings)
    #    avg_embedding = torch.mean(embeddings_tensor, dim=0)
    #    label_to_avg_embedding[label] = avg_embedding
    #    print("label_to_avg_embedding",label_to_avg_embedding[label].size())

    ## Calculate imagewithattributes_embeddings
    #label_to_imagewithattributes = {}
    #for label, avg_embedding in label_to_avg_embedding.items():
    #    avg_norm_embedding = F.normalize(avg_embedding, p=2, dim=0)
    #    imagewithattributes_embedding = torch.matmul(avg_norm_embedding, attribute_embeddings_tensor.t())
    #    label_to_imagewithattributes[label] = imagewithattributes_embedding
    #    print("label_to_imagewithattributes",label_to_imagewithattributes[label])

        
    for label, avg_imagewithattributes_embedding in label_to_avg_imagewithattributes.items():
        data = avg_imagewithattributes_embedding.detach().cpu().numpy()
        #print("data",data)
        #Custom nonlinear transformations: Combining logarithmic and exponential transformations
        custom_transformed_data = np.exp(np.log(data) * 3)
        plt.figure(figsize=(20, 7))    # Increase figure size to accommodate labels
        plt.bar(range(len(custom_transformed_data)), custom_transformed_data)
        label_name = label_mapping.get(label, f"Label {label}")
        plt.title(f"Attributes similarity scores for {label_name}",fontsize=20)
        plt.xlabel('Index of Attributes',fontsize=15)  # Reflects the total number of elements
        plt.ylabel('scores',fontsize=15)  # Compares each data's size
        plt.yticks([])  # Remove the tick labels from the y-axis
        plt.xticks(range(len(attributes)), range(len(attributes)), rotation='vertical')  # Use attribute names for x-ticks
        plt.tight_layout()  # Adjust layout to make room for x-labels
        #plt.savefig(f'{label_name}attributes.png', dpi=300, bbox_inches='tight')
        #plt.close()
        

    # Find the top values and their corresponding attributes for each label
    top_k = 1  # You can set this to however many top attributes you want
    label_to_top_attributes = {}
    for label, avg_imagewithattributes_embedding in label_to_avg_imagewithattributes.items():
        top_values, top_indices = torch.topk(avg_imagewithattributes_embedding, top_k)
        print("top_indices",top_indices)
        top_indices_list = top_indices.tolist()
        top_attributes = [attributes[i] for i in top_indices_list]
        label_to_top_attributes[label] = (top_values, top_attributes)
    
    if cfg['evaluate_with_novel_classes']:
        (_, _), (_, _), (_, _),  (_, _),_, novel_classes = read_split_data(json_file, ROOT)
        label_to_top_attributes = remap_keys_in_dict(label_to_top_attributes, novel_classes)

    return label_to_top_attributes

def get_umap(attribute_embeddings, test_loader):
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
    norm_image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    imagewithattributes_embeddings = torch.matmul(norm_image_embeddings, attribute_embeddings_tensor.t())
    print("imagewithattributes_embeddings",imagewithattributes_embeddings.size())
    random_state_value = random.randint(0, 10000)
    print("random_state_value",random_state_value)
    if with_zscore:
        #zero-mean and unit variance(z-score normalization)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(imagewithattributes_embeddings.cpu().detach().numpy())
        reducer = umap.UMAP(n_components=2,metric="cosine",random_state= random_state_value)
        reduced_embeddings = reducer.fit_transform(normalized_data)

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        # Create a discrete colormap
        unique_labels = torch.unique(image_labels)
        num_classes = len(unique_labels)
        colors = plt.cm.get_cmap('tab10', num_classes)  # 'tab10' has 10 colors, you can use other colormaps as well
        
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            c=image_labels.cpu().detach().numpy(), cmap=colors)
        
        # Create a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10) 
                for i in range(num_classes)]
        labels = [label_mapping.get(int(label), f"Label {label}") for label in unique_labels.numpy()]
        plt.legend(handles, labels, title="Labels")

        plt.title('UMAP projection of image with attributes embeddings')
        plt.savefig('umapwithzscore_labelname_fixpatches.png', dpi=300, bbox_inches='tight')
        plt.show()

        #print("normalized_data",normalized_data)
    # Apply UMAP to reduce to 2D
    if without_zscore:
        
        reducer = umap.UMAP(n_components=2,metric="cosine",random_state= random_state_value)
        reduced_embeddings = reducer.fit_transform(imagewithattributes_embeddings.cpu().detach().numpy())

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        # Create a discrete colormap
        unique_labels = torch.unique(image_labels)
        num_classes = len(unique_labels)
        colors = plt.cm.get_cmap('tab10', num_classes)  # 'tab10' has 10 colors, you can use other colormaps as well
        
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            c=image_labels.cpu().detach().numpy(), cmap=colors)
        
        # Create a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10) 
                for i in range(num_classes)]
        labels = [label_mapping.get(int(label), f"Label {label}") for label in unique_labels.numpy()]
        plt.legend(handles, labels, title="Labels")

        plt.title('UMAP projection of image with attributes embeddings')
        plt.savefig('umapwithoutzscore_labelname_fixpatches.png', dpi=300, bbox_inches='tight')
        plt.show()


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
        accuracy, eachclass_accuracy = evaluate_novel_classes(attributes_embeddings, all_test_loader)
        print(f"Accuracy: {accuracy}")
        print(f"eachclass_accuracy: {eachclass_accuracy}")
        # Additional suggestions to troubleshoot accuracy issues
        print(f"Total number of test samples: {len(all_test_loader.dataset)}")
        attribute=attribute_with_high_score(attributes_embeddings, all_test_loader,attributes)
        print(attribute)
        
    if cfg['umap']:
        get_umap(attributes_embeddings, all_test_loader)
    

    return _


if __name__ == '__main__':
    
    main(cfg)


