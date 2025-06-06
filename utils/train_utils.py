'''
evaluate zero-shot performance
'''
import copy
import time

# import open_clip

from dataset import FeatureDataset, OnlineScoreDataset
from utils.dataset_utils import *


def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def construct_attributes_save_path(cfg):

    if cfg['cluster_feature_method'] == 'random' or cfg['cluster_feature_method'] == 'kmeans':
        mahalanobis = False
    else:
        mahalanobis = cfg['mahalanobis']

    if not mahalanobis:
        return cfg['dataset'] + '_' + cfg['attributes'] + '_'  \
        + '_' + cfg['cluster_feature_method'] + '_' + str(cfg['num_attributes']) + '_' + str(cfg['reinit']) + '.txt'
    else:
        return cfg['dataset'] + '_' + cfg['attributes'] + '_'  \
        + '_' + cfg['cluster_feature_method'] + '_' + str(cfg['num_attributes']) + "_" + 'mahalanobis' + '_' + str(cfg['division_power']) + '_' + str(cfg['reinit']) + '.txt'

def get_model(cfg, model, input_dim, output_dim):

    if cfg['num_attributes'] == 'full':
        num_attributes = len(get_attributes(cfg))
    else:
        num_attributes = cfg['num_attributes']

    if model == ['linear', 'bn', 'linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.BatchNorm1d(num_attributes),
            nn.Linear(num_attributes, output_dim)
        )
    elif model == ['bn', 'linear']:
        model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, output_dim, bias=False),
        )
    elif model == ['linear', 'linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.Linear(num_attributes, output_dim)
        )
    elif model == ['linear']:
        model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )

    else:
        raise NotImplementedError

    return model


def get_feature_dataloader(cfg):

    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    elif cfg['model_type'] == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(cfg['model_size'], pretrained=cfg['openclip_pretrain'], device='cuda')
    else:
        raise NotImplementedError
    '''
    train_loader, test_loader = get_image_dataloader(cfg['dataset'], preprocess)

    train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
    test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')
    '''
    #for fungi
    if cfg['base_ratio'] == 1.0:
        base_train_loader,base_test_loader = get_image_dataloader(cfg['dataset'], preprocess)
        base_train_features = get_image_embeddings(cfg, cfg['dataset'], model, base_train_loader, 'base_train')
        base_test_features = get_image_embeddings(cfg, cfg['dataset'], model, base_test_loader, 'base_test')
        
        base_train_labels,base_test_labels = get_labels(cfg['dataset'])
        
        base_train_dataset = FeatureDataset(base_train_features, base_train_labels)
        base_test_dataset = FeatureDataset(base_test_features, base_test_labels)
        base_train_loader = DataLoader(base_train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        base_test_loader = DataLoader(base_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
        return base_train_loader,base_test_loader
    else:
        base_train_loader,base_test_loader, novel_test_loader,all_test_loader,all_train_loader = get_image_dataloader(cfg['dataset'], preprocess)
        base_train_features = get_image_embeddings(cfg, cfg['dataset'], model, base_train_loader, 'base_train')
        base_test_features = get_image_embeddings(cfg, cfg['dataset'], model, base_test_loader, 'base_test')
        novel_test_features = get_image_embeddings(cfg, cfg['dataset'], model, novel_test_loader, 'novel_test')
        all_test_features = get_image_embeddings(cfg, cfg['dataset'], model, all_test_loader, 'all_test')
        all_train_features = get_image_embeddings(cfg, cfg['dataset'], model, all_train_loader, 'all_train')
    ####
        base_train_labels,base_test_labels,novel_test_labels,test_images_labels,train_images_labels = get_labels(cfg['dataset'])
        base_train_dataset = FeatureDataset(base_train_features, base_train_labels)
        base_test_dataset = FeatureDataset(base_test_features, base_test_labels)
        novel_test_dataset = FeatureDataset(novel_test_features, novel_test_labels)
        all_test_dataset = FeatureDataset(all_test_features, test_images_labels)
        all_train_dataset = FeatureDataset(all_train_features, train_images_labels)
        
        base_train_loader = DataLoader(base_train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        base_test_loader = DataLoader(base_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
        novel_test_loader = DataLoader(novel_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
        all_test_loader = DataLoader(all_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
        all_train_loader = DataLoader(all_train_dataset, batch_size=cfg['batch_size'], shuffle=False)

        return base_train_loader,base_test_loader,novel_test_loader,all_test_loader,all_train_loader

def get_novel_test_feature_dataloader(cfg):
    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    elif cfg['model_type'] == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(cfg['model_size'], pretrained=cfg['openclip_pretrain'], device='cuda')
    else:
        raise NotImplementedError
    _,_, novel_test_loader,_,_ = get_image_dataloader(cfg['dataset'], preprocess)
    novel_test_features = get_image_embeddings(cfg, cfg['dataset'], model, novel_test_loader, 'novel_test')
    _,_,novel_test_labels,_,_ = get_labels(cfg['dataset'])
    novel_test_dataset = FeatureDataset(novel_test_features, novel_test_labels)
    novel_test_loader = DataLoader(novel_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
    return novel_test_loader

def get_all_test_feature_dataloader(cfg):
    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    elif cfg['model_type'] == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(cfg['model_size'], pretrained=cfg['openclip_pretrain'], device='cuda')
    else:
        raise NotImplementedError
    if cfg['base_ratio'] != 1.0:
        _,_, _,all_test_loader,_ = get_image_dataloader(cfg['dataset'], preprocess)
        _,_,_,test_images_labels,_ = get_labels(cfg['dataset'])
    else:
        _,all_test_loader = get_image_dataloader(cfg['dataset'], preprocess)
        _,test_images_labels = get_labels(cfg['dataset'])    
    all_test_features = get_image_embeddings(cfg, cfg['dataset'], model, all_test_loader, 'all_test')
    all_test_dataset = FeatureDataset(all_test_features, test_images_labels)
    all_test_loader = DataLoader(all_test_dataset, batch_size=cfg['batch_size'], shuffle=False)
    return all_test_loader

def get_score_dataloader(cfg, attribute_embeddings):

    if cfg['model_type'] == 'clip':
        model, preprocess = clip.load(cfg['model_size'])
    elif cfg['model_type'] == 'open_clip':
        model, _, preprocess = open_clip.create_model_and_transforms(cfg['model_size'], pretrained=cfg['openclip_pretrain'], device='cuda')
    else:
        raise NotImplementedError
    train_loader, test_loader = get_image_dataloader(cfg['dataset'], preprocess)
    print("Get Embeddings...")
    train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
    test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')
    if cfg['dataset'] == 'imagenet-animal':
        train_labels, test_labels = get_labels("imagenet")
        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        train_idxes = np.where((train_labels < 398) & (train_labels!=69))
        train_features = train_features[train_idxes]

        test_idxes = np.where((test_labels < 398) & (test_labels!=69))
        test_features = test_features[test_idxes]

    if cfg['dataset'] == 'waterbirds':
        train_labels, test_labels, train_group_array, test_group_array, train_selected_indices, test_selected_indices  = get_labels(cfg['dataset'])
        if len(train_labels) != len(train_features):
            train_features = train_features[train_selected_indices]
            test_features = test_features[test_selected_indices]
    else:
        train_labels, test_labels = get_labels(cfg['dataset'])


    print("Initializing Feature Dataset")
    train_feature_dataset = FeatureDataset(train_features, train_labels)
    test_feature_dataset = FeatureDataset(test_features, test_labels)
    train_loader = DataLoader(train_feature_dataset, batch_size=cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_feature_dataset, batch_size=cfg['batch_size'], shuffle=False)

    train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)
    test_scores = extract_concept_scores(test_loader, model, attribute_embeddings)

    if cfg['dataset'] == 'waterbirds':
        train_score_dataset = FeatureDataset(train_scores, train_labels, train_group_array)
        test_score_dataset = FeatureDataset(test_scores, test_labels, test_group_array)
    else:
        train_score_dataset = FeatureDataset(train_scores, train_labels)
        test_score_dataset = FeatureDataset(test_scores, test_labels)

    train_loader = DataLoader(train_score_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_score_dataset, batch_size=cfg['batch_size'], shuffle=False)

    return train_loader, test_loader

def calculate_worst_group_acc(predictions, labels, groups):
    st = time.time()
    comparison = predictions == labels
    worst_group_acc = 1
    for i in range(4):
        indices = torch.where(groups==i)
        acc = torch.sum(comparison[indices]) / len(indices[0])
        worst_group_acc = min(worst_group_acc, acc)
    et = time.time()
    return worst_group_acc


def train_model(cfg, epochs, model, train_loader, test_loader, regularizer=None, configs=None,use_voting=cfg['use_voting']):

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    loss_function = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_worst_group_acc = 0
    last_best_acc = None
    best_model = copy.deepcopy(model)

    no_break = False
    if epochs < 0:
        print("No Early Stopping")
        epochs = - epochs
        no_break = True

    for epoch in range(epochs):
        # train:
        total_hit = 0
        total_num = 0
        for idx, batch in enumerate(train_loader):
            s, t = batch[0], batch[1]
            s = s.float().cuda()
            t = t.long().cuda()
            output = model(s)
            loss = loss_function(output, t)

            if regularizer == 'mahalanobis':
                mahalanobis_loss = (mahalanobis_distance(model[0].weight/model[0].weight.data.norm(dim=-1, keepdim=True), configs['mu'].cuda(),
                        configs['sigma_inv'].cuda()) - configs['mean_distance']) / (configs['mean_distance']**cfg['division_power'])
                loss += torch.abs(mahalanobis_loss)

            elif regularizer == 'cosine':

                weight = model[0].weight/model[0].weight.data.norm(dim=-1, keepdim=True)
                loss += cfg['lambda'] * torch.sum((weight - configs['mu'].unsqueeze(0).cuda()) ** 2, dim=-1).mean()

            total_hit += torch.sum(t == output.argmax(-1))
            total_num += len(t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("after training")

        # test:
        with torch.no_grad():
            group_array = []
            predictions = []
            labels = []
            for idx, batch in enumerate(test_loader):
                s, t = batch[0], batch[1]
                s = s.float().cuda()
                output = model(s).cpu()
                #output = model(s)
                pred = torch.argmax(output, dim=-1)
                if len(batch) == 3:
                    group_array.append(batch[2])
                predictions.append(pred)
                labels.append(t)
            

            predictions = torch.cat(predictions)
            if len(group_array) > 0:
                group_array = torch.cat(group_array)

        labels = torch.cat(labels)
        #print("torch.cat(labels) labels:", labels)
        #print("predictions:", predictions)

        if use_voting:
            patches_per_image = 7
            num_images = len(predictions) // patches_per_image
            predictions = predictions.view(num_images, patches_per_image)    
            # torch.mode: Calculate the mode of each row
            voted_predictions = torch.mode(predictions, dim=1).values
            correct_labels = labels[patches_per_image-1::patches_per_image][:num_images] 
            acc = (torch.sum(voted_predictions == correct_labels) / len(correct_labels) * 100)
        else:
            acc = (torch.sum(predictions == labels) / len(predictions) * 100)

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}], Best accuracy: {best_acc}, Last accuracy: {acc}")
            #print(f"Epoch [{epoch}], Best accuracy:", best_acc.item(), "Last accuracy:", acc.item())

            sys.stdout.flush()

            if not no_break and (last_best_acc is not None and best_acc == last_best_acc):
                break
            last_best_acc = best_acc

    return best_model, best_acc

def mahalanobis_distance(x, mu, sigma_inv):
    x = x - mu.unsqueeze(0)
    return torch.diag(x @ sigma_inv @ x.T).mean()



def get_image_embeddings(cfg, dataset, model, loader, mode='train'):

    print ("Extract image features...")
    with torch.no_grad():
        features = []
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            (images, target) = batch[0], batch[1]
            # images: [batch_size, 3, 224, 224]
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            # [batch_size, 768]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # [batch_size, 768]
            features.append(image_features.cpu())
        features = torch.cat(features)
    features = np.array(features)
       

    return features



def extract_concept_scores(loader, model, attribute_embeddings):
    with torch.no_grad():
        scores = []

        for i, (image_features, target) in tqdm(enumerate(loader), total=len(loader)):
            image_features = image_features.cuda().float()
            # target = target.cuda()
            # image_features = model.encode_image(images).float()
            # image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ attribute_embeddings.float().T.cuda()
            scores.extend(logits.cpu().to(torch.float16).tolist())

    return scores




