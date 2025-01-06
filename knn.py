from utils.train_utils import *
from cluster import cluster
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


def get_auroc(train_loader, all_train_loader,test_loader, attribute_embeddings):
    test_image_embeddings = []
    test_image_labels = []
    train_image_embeddings = []
    train_image_labels = []
    base_train_image_embeddings = []
    base_train_image_labels = []

    # Extract image embeddings and tags from the loader
    for embeddings, labels in test_loader:
        test_image_embeddings.append(embeddings)
        test_image_labels.append(labels)
        
    for embeddings, labels in train_loader:
        base_train_image_embeddings.append(embeddings)
        base_train_image_labels.append(labels)
        
    for embeddings, labels in all_train_loader:
        train_image_embeddings.append(embeddings)
        train_image_labels.append(labels)
        
    test_image_embeddings = torch.cat(test_image_embeddings, dim=0)
    test_image_labels = torch.cat(test_image_labels, dim=0)
    test_image_embeddings = test_image_embeddings.float()
    
    base_train_image_embeddings = torch.cat(base_train_image_embeddings, dim=0)
    base_train_image_labels = torch.cat(base_train_image_labels, dim=0)
    base_train_image_embeddings = base_train_image_embeddings.float()
    
    train_image_embeddings = torch.cat(train_image_embeddings, dim=0)
    train_image_labels = torch.cat(train_image_labels, dim=0)
    train_image_embeddings = train_image_embeddings.float()

    attribute_embeddings_tensor = attribute_embeddings.clone().detach()
    attribute_embeddings_tensor = attribute_embeddings_tensor.float()
    
    x_train = torch.matmul(train_image_embeddings, attribute_embeddings_tensor.t()).numpy()
    x_test = torch.matmul(test_image_embeddings, attribute_embeddings_tensor.t()).numpy()

    y_train = np.array([1 if label in base_train_image_labels else 0 for label in train_image_labels])

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(x_train, y_train)

    y_test_pred = knn.predict(x_test)

    y_test = np.array([1 if label in base_train_image_labels else 0 for label in test_image_labels])

    roc_auc = roc_auc_score(y_test, y_test_pred)
    print(f"AUROC: {roc_auc}")
    
    
    
def main(cfg):
                    
    set_seed(cfg['seed'])
    print(cfg)    
    if cfg['cluster_feature_method'] == 'linear' and cfg['num_attributes'] != 'full':
        acc, model, attributes, attributes_embeddings = cluster(cfg)
        #print(attributes_embeddings)
    else:
        attributes, attributes_embeddings = cluster(cfg)        
    if cfg['knn-classifier'] and cfg['base_ratio'] != 1.0:    
        base_train_loader,_,_,all_test_loader ,all_train_loader= get_feature_dataloader(cfg)
        get_auroc(base_train_loader,all_train_loader, all_test_loader, attributes_embeddings)
        
    return None


if __name__ == '__main__':
    
    main(cfg)
