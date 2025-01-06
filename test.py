from utils.train_utils import *
from cluster import cluster
from utils.rat1 import *
from utils.attribute_similarity_scores import *

def main(cfg):

    set_seed(cfg['seed'])
    print(cfg)

    if cfg['cluster_feature_method'] == 'linear' and cfg['num_attributes'] != 'full':
        acc, model, attributes, attributes_embeddings = cluster(cfg)
        #print(attributes_embeddings)
    else:
        attributes, attributes_embeddings = cluster(cfg)
    
    if cfg['evaluate_with_novel_classes']:
        if cfg['base_ratio'] != 1.0:
            novel_test_loader =get_novel_test_feature_dataloader(cfg)
            accuracy, eachclass_accuracy = evaluate_on_classes(cfg, attributes_embeddings, novel_test_loader)
            print(f"R@1 Accuracy: {accuracy}")
            print(f"R@1 eachclass_accuracy: {eachclass_accuracy}")
            print(f"Total number of test samples: {len(novel_test_loader.dataset)}")
        else:
            print("Please set base_ratio to a value less than 1.0 to evaluate with novel classes.")
    else:
        all_test_loader = get_all_test_feature_dataloader(cfg)
        accuracy, eachclass_accuracy = evaluate_on_classes(cfg, attributes_embeddings, all_test_loader)
        print(f"R@1 Accuracy: {accuracy}")
        print(f"R@1 eachclass_accuracy: {eachclass_accuracy}")
        print(f"Total number of test samples: {len(all_test_loader.dataset)}")
        
    if cfg['umap'] and not cfg['evaluate_with_novel_classes']:
        get_umap(cfg, attributes_embeddings, all_test_loader)
        
         
    if cfg['attribute_similarity_scores'] and not cfg['evaluate_with_novel_classes']:
        #Randomly select one image from each class,and calculate the similarity score with each attribute
        attribute_with_random_image_score(cfg, attributes_embeddings, all_test_loader,attributes)
        #Select all images from each class,and calculate the similarity score with each attribute
        attribute_with_all_images_score(cfg, attributes_embeddings, all_test_loader,attributes) 

    return None


if __name__ == '__main__':
    
    main(cfg)


