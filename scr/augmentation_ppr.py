import random
import os
import torch
import numpy as np
from pprfile import *
import scipy


def calc_ppr(args,adj_dict):
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_type=args.target_type
    ppr={}
    for k in net_schema[target_type]:
        ppr_file=args.log_dir+'/ppr_'+target_type+'_'+k+'.pt'
        if os.path.exists(ppr_file):
            if args.dataset=='imdb' and k=='a':
                ppr[k]=torch.load(ppr_file)
            else:
                ppr[k]=torch.load(ppr_file).to_dense()

        else:    
            adj=adj_dict[target_type][k].clone().to_dense()
            if k=='citing_p' :
                adj_T=adj_dict[target_type]['cited_p'].clone().to_dense()
            elif k=='cited_p' :
                continue          
            else:
                adj_T=adj_dict[k][target_type].clone().to_dense()  
            A = torch.zeros(adj.shape[0]+adj.shape[1], adj.shape[0]+adj.shape[1]).to(args.device)
            adj[adj!=0]=1
            adj_T[adj_T!=0]=1
            if k=='citing_p':
                A = adj             
            else:
                A[:adj.shape[0],adj.shape[0]:] = adj
                A[adj.shape[0]:,:adj.shape[0]] = adj_T

            if args.dataset=='imdb' and k=='a':
                train_idx=np.arange(A.shape[0])
                A = scipy.sparse.csr_matrix(np.matrix(A.cpu()))
                PPRM = topk_ppr_matrix(A, alpha=0.15 , eps=1e-4 , idx=train_idx, topk=0, normalization='sym')
            else:
                PPRM=PPR(args,A)
            if k=='citing_p':
                torch.save(PPRM.to_sparse_coo(),ppr_file) 
                ppr[k]=(PPRM) 
            elif args.dataset=='imdb' and k=='a':
                torch.save(PPRM[:adj.shape[0],adj.shape[0]:],ppr_file) 
                ppr[k]=(PPRM[:adj.shape[0],adj.shape[0]:])                    
            else:
                torch.save(PPRM[:adj.shape[0],adj.shape[0]:].to_sparse_coo(),ppr_file) 
                ppr[k]=(PPRM[:adj.shape[0],adj.shape[0]:]) 

    # meta_path
    for meta in args.meta_path:
        k = meta[1]
        kk = meta[2]
        kkk = meta[3]
        
        ppr_file=args.log_dir+'/ppr_'+target_type+'_'+k+kk+kkk+'.pt'
        if os.path.exists(ppr_file):
            ppr[k+kk+kkk]=(torch.load(ppr_file)).to_dense()
            
        else:    
            adj1=adj_dict[target_type][k].clone().to_dense()
            adj2=adj_dict[k][kk].clone().to_dense()
            adj3=adj_dict[kk][kkk].clone().to_dense()
            adj1[adj1!=0]=1
            adj2[adj2!=0]=1
            adj3[adj3!=0]=1

            adj=adj1 @ adj2
            adj=adj @ adj3
            A = torch.zeros(adj.shape[0]+adj.shape[1], adj.shape[0]+adj.shape[1])
            A[:adj.shape[0],adj.shape[0]:] = adj
            A[adj.shape[0]:,:adj.shape[0]] = torch.t(adj)
            PPRM=PPR(args,A)
            torch.save(PPRM[:adj.shape[0],adj.shape[0]:].to_sparse_coo(),ppr_file)
            ppr[k+kk+kkk]=(PPRM[:adj.shape[0],adj.shape[0]:]) 
    return ppr


def PPR(args,A):
    pagerank_prob=0.85
    pr_prob = 1 - pagerank_prob
    A_hat   = A.to(args.device) + torch.eye(A.size(0)).to(args.device) 
    D       = torch.diag(torch.sum(A_hat,1))
    D       = D.inverse().sqrt()
    A_hat   = torch.mm(torch.mm(D, A_hat), D)
    Pi = pr_prob * ((torch.eye(A.size(0)).to(args.device) - (1 - pr_prob) * A_hat).inverse())
    Pi = Pi.cpu()
 
    return Pi 




def augmentation(args,  ppr, ft_dict_ori, adj_dict_ori, true_label, idx_train):
    adj_dict={}
    for k in adj_dict_ori:
        adj_dict[k]={}
        for kk in adj_dict_ori[k]:
            adj_dict[k][kk]=adj_dict_ori[k][kk].clone()
    ft_dict={}
    for k in ft_dict_ori:
        ft_dict[k]=ft_dict_ori[k].clone()


    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_type = args.target_type

    all_pseudo_idx=None
    true_label_aug =  true_label.clone()
    idx_train_aug = idx_train.clone()

    minor_cla_num = int(np.floor(args.num_class/2))
    for cla in range(minor_cla_num):
        minor_class_idx =args.num_class-minor_cla_num+cla
        generate_num = args.major_num-args.minor_num
        minor_idx = idx_train[true_label[idx_train] == minor_class_idx]
        # adj 
        for k in net_schema[target_type]:
            if k =='cited_p':
                continue
            A = adj_dict[target_type][k].clone().to_dense()

            degree_max = max(torch.sum(A[minor_idx]!=0, dim=1).cpu()) 
            degree = torch.sum(A[minor_idx]!=0, dim=1).cpu()

            if torch.is_tensor(ppr[k]):
                influence = torch.sum(ppr[k][minor_idx.cpu()], dim=0)
            else:
                influence = torch.sum(torch.tensor(ppr[k][minor_idx.cpu()].todense()), dim=0)

            if args.meta_path!=[]:
                for meta in args.meta_path:
                    if k == meta[-1]:
                        key=''.join(meta[1:])
                        influence += torch.sum(ppr[key][minor_idx], dim=0)
            
            max_node=(influence!=0).sum()
            if args.neighbor_scale<1:
                sample_num=max([degree_max.item(), int(args.neighbor_scale*max_node)])
            else:
                sample_num=min([len(influence), int(degree_max.item()*args.neighbor_scale)])
            _, candidate = torch.topk(influence, k=sample_num, largest=True)
 
            
            probability = torch.zeros(influence.shape[0])
            probability[candidate] = 1



            A_new = torch.zeros(generate_num, influence.shape[0]).to(args.device)

            for i in range(generate_num):
                degree_new=(np.random.choice(degree, size=1, replace=True).item())
                if degree_new!=0:
                    idx = torch.multinomial(probability, num_samples= degree_new)
                    A_new[i, idx]=1


            if k=='citing_p' :
                A[A!=0]=1
                if cla>0:
                    A_new=(torch.cat([A_new, torch.zeros(generate_num*cla, generate_num*cla).to(args.device)], dim=1))
                A=torch.cat([A,A_new], dim=0)
                A_new=torch.t(torch.cat([A_new, torch.zeros(generate_num, generate_num).to(args.device)], dim=1))
                A=torch.cat([A,A_new], dim=1)

                degree = 1/torch.sum(A,1)
                degree[torch.isinf(degree)]=0
                D      = torch.diag(degree)
                A  = torch.mm(D, A)
                adj_dict[target_type][k]= A.to_sparse_coo()
                adj_dict[target_type]['cited_p']= A.to_sparse_coo()

            else:
                degree = 1/torch.sum(A_new,1)
                degree[torch.isinf(degree)]=0
                D      = torch.diag(degree)
                A_new  = torch.mm(D, A_new)
                A=torch.cat([A,A_new], dim=0)
                adj_dict[target_type][k]= A.to_sparse_coo()

                A[A!=0]=1
                A_T=torch.t(A)
                degree = 1/torch.sum(A_T,1)
                degree[torch.isinf(degree)]=0
                D       = torch.diag(degree)
                A_T = torch.mm(D, A_T)
                adj_dict[k][target_type]= A_T.to_sparse_coo()

        # label  
        
        pseudo_idx = torch.tensor(np.arange(true_label_aug.shape[0], true_label_aug.shape[0]+generate_num)).long().to(args.device)
        if all_pseudo_idx == None:
            all_pseudo_idx = pseudo_idx.clone()
        else:
            all_pseudo_idx = torch.cat([all_pseudo_idx,pseudo_idx], dim=0)       
        pseudo_label=minor_class_idx*torch.ones(generate_num, dtype=int).to(args.device)
        true_label_aug = torch.cat([true_label_aug,pseudo_label], dim=0)

        # feature
        feat = ft_dict[target_type][minor_idx]  
        smote_num=2
        feat_smote = torch.zeros(generate_num, feat.shape[1]).to(args.device)
        for i in range(generate_num):
            idx=np.random.choice(minor_idx.shape[0], size=smote_num, replace=False)
            feat_smote[i]=torch.mean(feat[idx], dim=0)

        ft_dict[target_type] = torch.cat([ft_dict[target_type],feat_smote], dim=0)


    all_pseudo_idx = np.array(all_pseudo_idx.cpu())
    random.shuffle(all_pseudo_idx)
    return ft_dict, adj_dict, true_label_aug, all_pseudo_idx