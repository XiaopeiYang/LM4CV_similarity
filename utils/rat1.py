import torch
import torch.nn.functional as F
from .train_utils import read_split_data
label_mapping = {
    0: "Caspofungin",
    1: "Carbendazim",
    2: "Mycelium",
    3: "Germination Inhibition",
    4: "GWT1",
    5: "Tebuconazole"
}

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
#check whether normliazed  
def is_normalized(tensor, p=2, dim=1, atol=1e-5):
    norms = torch.norm(tensor, p=p, dim=dim)
    return torch.allclose(norms, torch.ones_like(norms), atol=atol)

def calculate_cosine_similarity(embeddings):

    # Calculate cosine similarity matrix
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    #print("norm_embedding",norm_embeddings.size())
    similarity_matrix = torch.einsum('ik,jk->ij', norm_embeddings, norm_embeddings)
    #print("similarity_matrix",similarity_matrix.size())
    return similarity_matrix

def image_retrieval(similarity_matrix, top_k=1):
    # Image retrieval based on similarity to find the most similar image
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k + 1, dim=1, largest=True, sorted=True)
    retrieval_results = top_k_indices[:, 1:]  # exclude oneself
    #("retrieval_results",retrieval_results.size())
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

def calculate_eachclass_accuracy(cfg, retrieval_results, labels):
    # Calculate the accuracy for each class
    class_correct = {}
    class_total = {}
    
    for idx, retrieved_indices in enumerate(retrieval_results):
        label = labels[idx].item()
        #print("label",label)
        #print("labels[retrieved_indices]",labels[retrieved_indices])
        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0
        
        if label in labels[retrieved_indices]:
            class_correct[label] += 1
        class_total[label] += 1
    class_accuracy = {label: class_correct[label] / class_total[label] for label in class_correct}
    if cfg['evaluate_with_novel_classes']:
        (_, _), (_, _), (_, _),  (_, _),(_,_),_, novel_classes = read_split_data(cfg['json_file'], cfg['ROOT'])
        class_accuracy = remap_keys_in_dict(class_accuracy, novel_classes)
    return class_accuracy

def evaluate_on_classes(cfg, attribute_embeddings, test_loader):
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
    eachclass_accuracy = calculate_eachclass_accuracy(cfg, retrieval_results, image_labels)
    # Check some of the embeddings and labels to verify data integrity
    

    return accuracy, eachclass_accuracy


import matplotlib.pyplot as plt
import umap
import random
import os

from sklearn.preprocessing import StandardScaler
def get_umap(cfg, attribute_embeddings, test_loader):
    image_embeddings = []
    image_labels = []

    # Extract image embeddings and tags from the loader
    for embeddings, labels in test_loader:
        image_embeddings.append(embeddings)
        image_labels.append(labels)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    image_labels = torch.cat(image_labels, dim=0)
    image_embeddings = image_embeddings.float()


    # Map image_embeddings to attribute_embeddings space
    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    norm_image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
    imagewithattributes_embeddings = torch.matmul(norm_image_embeddings, attribute_embeddings_tensor.t())
    random_state_value = random.randint(0, 10000)
  
    if cfg['with_zscore']:
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

        plt.title('UMAP projection of image similarity')
        file_path = './umap/umapwithnormalization.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
        if os.path.exists(file_path):
            print(f"File saved successfully: {file_path}")
        else:
            print(f"Failed to save file: {file_path}")

        #print("normalized_data",normalized_data)
    # Apply UMAP to reduce to 2D
    if cfg['without_zscore']:
        
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
        
        plt.title('UMAP projection of image similarity')
        file_path = './umap/umapwithoutnormalization.png'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
        if os.path.exists(file_path):
            print(f"File saved successfully: {file_path}")
        else:
            print(f"Failed to save file: {file_path}")