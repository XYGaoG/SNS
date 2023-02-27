import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from load_data import *
from evaluate import *
from params import *
from module.att_hgcn import ATT_HGCN
import warnings
import csv
from augmentation_ppr import *
from constraints import *

warnings.filterwarnings('ignore')

def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_input[0].data

def feature_update(args, ft_dict_ori, true_label, idx_train):

    ft_dict=ft_dict_ori.clone()
    generate_num = args.major_num-args.minor_num
    minor_cla_num = int(np.floor(args.num_class/2))
    for cla in range(minor_cla_num):
        minor_class_idx =args.num_class-minor_cla_num+cla
        minor_idx = idx_train[true_label[idx_train] == minor_class_idx]

        # feature
        feat = ft_dict[minor_idx]  
        smote_num=2
        feat_smote = torch.zeros(generate_num, feat.shape[1]).to(args.device)
        for i in range(generate_num):
            idx=np.random.choice(minor_idx.shape[0], size=smote_num, replace=False)

            node_dim = saliency.shape[1]
            saliency_dst = saliency[minor_idx[idx[0]]].abs()
            saliency_dst += 1e-10
            saliency_dst /= torch.sum(saliency_dst)

            K = int(node_dim * 0.1)
            mask_idx = torch.multinomial(saliency_dst, K)
            feat_smote[i]=torch.mean(feat[idx], dim=0)
            feat_smote[i][mask_idx]=feat[idx[0]][mask_idx]

        ft_dict = torch.cat([ft_dict,feat_smote], dim=0)

    return ft_dict 


def train(args, data):
    target_type=args.target_type
    ft_dict, adj_dict, label_aug, ft_dict_aug, adj_dict_aug = data

    label_target = label_aug[target_type]
    true_label = label_target[0]
    idx_train = label_target[1]
    idx_val = label_target[2]
    idx_test = label_target[3]
    pseudo_idx = label_target[4]

    layer_shape = []
    input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
    hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in args.hidden_dim]
    output_layer_shape = dict.fromkeys(ft_dict.keys(), args.num_class)

    layer_shape.append(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.append(output_layer_shape)


    model = ATT_HGCN(
        net_schema=net_schema,
        layer_shape=layer_shape,
        label_keys=list(label.keys()),
        type_fusion=args.type_fusion,
        type_att_size=args.type_att_size,
    )
    if args.cuda and torch.cuda.is_available():
        model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    stopper = EarlyStopping(args)
    bceloss=torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        model.hgc1.hete_agg[target_type].w_self.register_full_backward_hook(backward_hook)

        ft_dict_aug[target_type].requires_grad = True 
        logits, embd1, embd2, attention_dict = model(ft_dict_aug, adj_dict_aug)
        logits = F.log_softmax(logits[target_type], dim=1)
        idx_train_fin=torch.cat([idx_train,torch.tensor(pseudo_idx).to(args.device)])
        loss_cla = F.nll_loss(logits[idx_train_fin], true_label[idx_train_fin]) 
    
        
        semantic_embd = semantic_cal_with_target_node(args, embd1, net_schema, adj_dict_aug)
        prototype = prototype_cal(args, embd1, semantic_embd, true_label, idx_train, pseudo_idx)
        intra_loss = class_constraint(args, embd1, semantic_embd, prototype, true_label, idx_train, pseudo_idx)

        link_loss = link_constraint(args, embd2, net_schema, adj_dict_aug, pseudo_idx, bceloss)

        loss_train = loss_cla+args.alpha*link_loss+args.beta*intra_loss
     
        loss_train.backward()
        optimizer.step()
        train_micro_f1, train_macro_f1, train_bacc = score(logits[idx_train], true_label[idx_train])


        model.eval()
        with torch.no_grad():
            logits, _, _,_ = model(ft_dict, adj_dict)
            logits = F.log_softmax(logits[target_type], dim=1)            
            val_micro_f1, val_macro_f1, val_bacc  = score(logits[idx_val], true_label[idx_val])
            test_micro_f1, test_macro_f1, test_bacc = score(logits[idx_test], true_label[idx_test])
            early_stop, counter, best_val_macro_f1 = stopper.step(val_macro_f1, model)
            print("Epoch {:d} | Train Loss {:.4f} | Train Macro f1 {:.4f} | Val Macro f1 {:.4f} | Test Macro f1 {:.4f}  | # Training sample  {:.0f} | Counter  {:.0f} | Best Val Macro f1 {:.4f}".format(
                    epoch + 1, loss_train.item(), train_macro_f1, val_macro_f1, test_macro_f1, len(idx_train_fin), counter, best_val_macro_f1))

        if early_stop:
            break

        if epoch>-1:
            ft_dict[target_type].requires_grad = False 
            ft_dict_aug[target_type] = feature_update(args, ft_dict[target_type], true_label, idx_train)

    stopper.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        logits, _, _,_ = model(ft_dict, adj_dict)
        logits = F.log_softmax(logits[target_type], dim=1)            
        test_micro_f1, test_macro_f1, test_bacc = score(logits[idx_test], true_label[idx_test])    
    print("Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(test_micro_f1, test_macro_f1))

    return test_micro_f1, test_macro_f1, test_bacc



if __name__ == '__main__':

    # parameters
    args = set_params()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device("cpu")
    args.device=device

    # data
    label, ft_dict, adj_dict = load_data(args, args.dataset, args.train_percent)   
    args.num_class = np.unique(label[args.target_type][0]).shape[0]
    if args.cuda and torch.cuda.is_available():
        for k in ft_dict:
            ft_dict[k] = ft_dict[k].to(args.device)
        for k in adj_dict:
            for kk in adj_dict[k]:
                adj_dict[k][kk] = adj_dict[k][kk].to(args.device)
        for k in label:
            for i in range(len(label[k])):
                label[k][i] = label[k][i].to(args.device)


    label_target = label[args.target_type]
    true_label = label_target[0]
    idx_train = label_target[1]
    idx_val = label_target[2]
    idx_test = label_target[3]

    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])    
    set_random_seed(args.seed)

    ppr = calc_ppr(args, adj_dict)
    ft_dict_aug_ori, adj_dict_aug, true_label_aug, pseudo_idx = augmentation(args, ppr, ft_dict, adj_dict, true_label, idx_train) 
    
    # training
    results_mic , results_mac, results_bacc = [],[],[]
    for n in range(args.repeat_num):
        args.seed += 1
        set_random_seed(args.seed)

        ft_dict_aug={}
        for k in ft_dict_aug_ori:
            ft_dict_aug[k]=ft_dict_aug_ori[k].clone()
        label_aug ={}
        label_aug[args.target_type] = [true_label_aug, idx_train, idx_val, idx_test, pseudo_idx]
        data_aug = [ft_dict, adj_dict, label_aug, ft_dict_aug, adj_dict_aug]

        test_micro_f1, test_macro_f1, test_bacc = train(args, data_aug)

        write_record=  ([test_macro_f1*100, test_micro_f1*100, test_bacc*100])
        with open(args.csv_file ,'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(write_record)

        results_mic.append(test_micro_f1)
        results_mac.append(test_macro_f1)
        results_bacc.append(test_bacc)
    record(args, results_mic,results_mac,results_bacc)
