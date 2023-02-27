import torch
import torch.nn.functional as F
import numpy as np



def class_constraint(args, embd, semantic_embd, prototype, true_label, idx_train, pseudo_idx):
    all_pseudo_idx= torch.tensor(pseudo_idx).to(args.device)
    cla_mask={}
    for cla in range(args.num_class):
        cla_mask[cla]=all_pseudo_idx[true_label[all_pseudo_idx]==cla]
    minor_cla_num = int(np.floor(args.num_class/2))
    
    cl_loss=0
    num=0
    for k in range(2):
        for cla in range(minor_cla_num):
            minor_class_idx = args.num_class-minor_cla_num+cla
            if k ==0: # target_type 
                traget_group=embd[args.target_type][cla_mask[minor_class_idx],:] # pseudo node
                positive = prototype[args.target_type][minor_class_idx]
                negative = torch.zeros(args.num_class, positive.shape[0])
                for cla2 in range(args.num_class):
                    negative[cla2]=prototype[args.target_type][cla2]

            else: #semantic 
                traget_group=semantic_embd[cla_mask[minor_class_idx],:]
                positive = prototype['semantic'][minor_class_idx]
                negative = torch.zeros(args.num_class, positive.shape[0])
                for cla2 in range(args.num_class):
                    negative[cla2]=prototype['semantic'][cla2]

            positive = torch.reshape(positive,(1,-1))
            negative = negative.to(args.device)

            traget_group = F.normalize(traget_group, dim=1)
            positive = F.normalize(positive, dim=1)               
            negative = F.normalize(negative, dim=1)

            pos_score = torch.matmul(traget_group, positive.transpose(0, 1))
            pos_score = torch.exp(pos_score / args.temperature).sum(dim=1)
            nag_score = torch.matmul(traget_group, negative.transpose(0, 1))
            nag_score = torch.exp(nag_score / args.temperature).sum(dim=1)
            cl_loss += torch.mean(-torch.log(pos_score / nag_score))
            num+=1
    intra_loss = cl_loss/num
    return intra_loss 




def link_constraint(args, embd, net_schema, adj_dict_aug, pseudo_idx, bceloss):
    link_loss=0.

    for k in net_schema[args.target_type]:
        if k =='citing_p':
            similarity=((embd[args.target_type]@embd[args.target_type].T))
            pseudo_adj=torch.tensor(adj_dict_aug[args.target_type][k].to_dense()!=0, dtype=torch.float32)   
            neighbor_mask=(torch.sum(pseudo_adj[pseudo_idx,:], dim=0)!=0)
            link_loss += bceloss(similarity[:,neighbor_mask].reshape(1,-1),pseudo_adj[:,neighbor_mask].reshape(1,-1))
        elif k =='cited_p':
            continue
        else:
            similarity=torch.mm(embd[args.target_type],embd[k].t())
            pseudo_adj=torch.tensor(adj_dict_aug[args.target_type][k].to_dense()!=0, dtype=torch.float32)          
            neighbor_mask=(torch.sum(pseudo_adj[pseudo_idx,:], dim=0)!=0)
            link_loss += bceloss(similarity[:,neighbor_mask].reshape(1,-1),pseudo_adj[:,neighbor_mask].reshape(1,-1))
    return link_loss


def inter_domin_contrast(args, prototype):

    negative = torch.zeros(args.num_class*2, prototype['semantic'][0].shape[0])
    num=0
    for cla in range(args.num_class): 
        negative[num]=prototype[args.target_type][cla]
        num+=1 
        negative[num]=prototype['semantic'][cla]
        num+=1
    cl_loss=0
    num=0
    for cla in range(args.num_class):
        for k in range(2):
            if k ==0: # target
                traget = prototype[args.target_type][cla]
                positive = prototype['semantic'][cla]
            else: #semantic
                traget =  prototype['semantic'][cla]
                positive = prototype[args.target_type][cla]

            traget = torch.reshape(traget,(1,-1))
            positive = torch.reshape(positive,(1,-1))
            negative = negative.to(args.device)

            traget = F.normalize(traget, dim=1)
            positive = F.normalize(positive, dim=1)               
            negative = F.normalize(negative, dim=1)

            pos_score = torch.matmul(traget, positive.transpose(0, 1))
            pos_score = torch.exp(pos_score / args.temperature).sum(dim=1)
            nag_score = torch.matmul(traget, negative.transpose(0, 1))
            nag_score = torch.exp(nag_score / args.temperature).sum(dim=1)
            cl_loss += torch.mean(-torch.log(pos_score / nag_score))
            num+=1
    inter_loss = cl_loss/num
    return inter_loss 




def semantic_cal_with_target_node(args, embd, net_schema, adj_dict):
    semantic_cal_buf={}
    for k in net_schema[args.target_type]:
        if k =='citing_p' or k =='cited_p':
            semantic_cal_buf[k] = torch.spmm(adj_dict[args.target_type][k], embd['p'])
        else:
            semantic_cal_buf[k] = torch.spmm(adj_dict[args.target_type][k], embd[k])

    num=1
    semantic_embd=embd[args.target_type]
    for k in net_schema[args.target_type]:
        semantic_embd+=semantic_cal_buf[k]
        num+=1
    return semantic_embd/num



def prototype_cal(args, embd, semantic_embd, true_label, idx_train, pseudo_idx):
    all_train_idx=idx_train 

    cla_mask={}
    for cla in range(args.num_class):
        cla_mask[cla]=all_train_idx[true_label[all_train_idx]==cla]

    prototype={}
    for domain in range(2): #target and semantic
        if domain==0:
            prototype[args.target_type]={}
            for cla in range(args.num_class):
                prototype[args.target_type][cla]=torch.mean(embd[args.target_type][cla_mask[cla],:], dim=0)
        else:
            prototype['semantic']={}
            for cla in range(args.num_class):
                prototype['semantic'][cla]=torch.mean(semantic_embd[cla_mask[cla],:], dim=0)
    return prototype 