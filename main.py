from utils.train_utils import *
from cluster import cluster

def main(cfg):

    set_seed(cfg['seed'])
    print(cfg)

    if cfg['cluster_feature_method'] == 'linear' and cfg['num_attributes'] != 'full':
        acc, model, attributes, attributes_embeddings = cluster(cfg)
        #print(attributes_embeddings)
    else:
        attributes, attributes_embeddings = cluster(cfg)
    if cfg['reinit']  and cfg['num_attributes'] != 'full':
        #print("cfg['reinit']  and cfg['num_attributes'] != 'full'")
        assert cfg['cluster_feature_method'] == 'linear'
        if cfg['base_ratio'] == 1.0:
            base_train_loader, base_test_loader = get_feature_dataloader(cfg)
        else:
            base_train_loader, base_test_loader,_, _, _ = get_feature_dataloader(cfg)
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(dim=-1, keepdim=True)
        for param in model[0].parameters():
            param.requires_grad = False
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, base_train_loader, base_test_loader)
    
    else:
        model = get_model(cfg, cfg['score_model'], input_dim=len(attributes), output_dim=get_output_dim(cfg['dataset']))
        score_train_loader, score_test_loader = get_score_dataloader(cfg, attributes_embeddings)
        best_model, best_acc = train_model(cfg, cfg['epochs'], model, score_train_loader, score_test_loader)

    return best_model, best_acc


if __name__ == '__main__':
    
    main(cfg)


