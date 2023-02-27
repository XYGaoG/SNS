import numpy as np
import scipy.sparse as sp
import torch
import pickle
import os


def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent = (1.0 - train_percent) / 2
    idx_train = torch.LongTensor(rand_idx[int(label_shape * 0.0): int(label_shape * train_percent)])
    idx_val = torch.LongTensor(
        rand_idx[int(label_shape * train_percent): int(label_shape * (train_percent + val_percent))])
    idx_test = torch.LongTensor(rand_idx[int(label_shape * (train_percent + val_percent)): int(label_shape * 1.0)])
    return idx_train, idx_val, idx_test

def train_val_test_split_imbalance(args,label_shape, train_percent, label, im_ratio):
    num_classes = np.unique(label).shape[0]
    idx_set, rest_indices, train_indices = [], [], []
    for i in range(num_classes):
        idx_set.append([])
        rest_indices.append([])
        train_indices.append([])

    rand_idx = np.random.permutation(label_shape)


    major_num = int(label_shape * train_percent / num_classes /10)*10
    minor_num = int(major_num * im_ratio)

    for idx in rand_idx:
        idx_set[label[idx]].append(idx)

    for i in range(num_classes):
        train_indices[i]=idx_set[i][:major_num]
        rest_indices[i]=idx_set[i][major_num:]

    if im_ratio !=1:
        minor_cla_num = int(np.floor(num_classes/2))
        for cla in range(minor_cla_num):
            train_indices[-cla-1]=train_indices[-cla-1][:minor_num]

    rest_indices = np.concatenate(rest_indices)
    train_indices = np.concatenate(train_indices)   
    np.random.shuffle(rest_indices)
    np.random.shuffle(train_indices)

    idx_train=torch.LongTensor(train_indices)
    idx_val = torch.LongTensor(rest_indices[: int( rest_indices.shape[0] /2)])
    idx_test = torch.LongTensor(rest_indices[int( rest_indices.shape[0] /2) :])
    return idx_train, idx_val, idx_test

def load_odbmag_4017(args, train_percent):
    path='../data/mag/'
    feats = np.load(path+'feats.npz', allow_pickle=True)
    p_ft = feats['p_ft']
    a_ft = feats['a_ft']
    i_ft = feats['i_ft']
    f_ft = feats['f_ft']

    ft_dict = {}
    ft_dict['p'] = torch.FloatTensor(p_ft)
    ft_dict['a'] = torch.FloatTensor(a_ft)
    ft_dict['i'] = torch.FloatTensor(i_ft)
    ft_dict['f'] = torch.FloatTensor(f_ft)

    p_label = np.load(path+'p_label.npy', allow_pickle=True)
    p_label = torch.LongTensor(p_label)

    file_name = path+'p_label_im_'+str(args.im_ratio)+'.pkl'
    if os.path.exists(file_name):
        with open(file_name, "rb+") as f:
            label = pickle.load(f)
    else:
        idx_train_p, idx_val_p, idx_test_p = train_val_test_split_imbalance(args, p_label.shape[0], train_percent, p_label, args.im_ratio)
        label = {}
        label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]
        with open(file_name, "wb") as f:
            pickle.dump(label, f)

    args.major_num = int(p_label.shape[0] * train_percent / np.unique(p_label).shape[0] /10)*10
    args.minor_num = int(args.major_num * args.im_ratio)


    sp_a_i = sp.load_npz(path+'norm_sp_a_i.npz')
    sp_i_a = sp.load_npz(path+'norm_sp_i_a.npz')
    sp_a_p = sp.load_npz(path+'norm_sp_a_p.npz')
    sp_p_a = sp.load_npz(path+'norm_sp_p_a.npz')
    sp_p_f = sp.load_npz(path+'norm_sp_p_f.npz')
    sp_f_p = sp.load_npz(path+'norm_sp_f_p.npz')
    sp_p_cp = sp.load_npz(path+'norm_sp_p_cp.npz')
    sp_cp_p = sp.load_npz(path+'norm_sp_cp_p.npz')

    adj_dict = {'p': {}, 'a': {}, 'i': {}, 'f': {}}
    adj_dict['a']['i'] = sp_coo_2_sp_tensor(sp_a_i.tocoo())
    adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp_a_p.tocoo())
    adj_dict['i']['a'] = sp_coo_2_sp_tensor(sp_i_a.tocoo())
    adj_dict['f']['p'] = sp_coo_2_sp_tensor(sp_f_p.tocoo())
    adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp_p_a.tocoo())
    adj_dict['p']['f'] = sp_coo_2_sp_tensor(sp_p_f.tocoo())
    adj_dict['p']['citing_p'] = sp_coo_2_sp_tensor(sp_p_cp.tocoo())
    adj_dict['p']['cited_p'] = sp_coo_2_sp_tensor(sp_cp_p.tocoo())

    return label, ft_dict, adj_dict

def load_imdb_3228(args, train_percent):
    data_path = '../data/imdb/imdb3228.pkl'

    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)

        m_label = label['m'][0]
        path='../data/imdb/'
        file_name = path+'m_label_im_'+str(args.im_ratio)+'.pkl'
        if os.path.exists(file_name):
            with open(file_name, "rb+") as f:
                label = pickle.load(f)
        else:
            idx_train_m, idx_val_m, idx_test_m = train_val_test_split_imbalance(args,m_label.shape[0], train_percent, m_label, args.im_ratio)
            label = {}
            label['m'] = [m_label, idx_train_m, idx_val_m, idx_test_m]
            with open(file_name, "wb") as f:
                pickle.dump(label, f)

        args.major_num = int(m_label.shape[0] * train_percent / np.unique(m_label).shape[0] /10)*10
        args.minor_num = int(args.major_num * args.im_ratio)

        adj_dict['m']['a'] = adj_dict['m']['a'].to_sparse()
        adj_dict['m']['u'] = adj_dict['m']['u'].to_sparse()
        adj_dict['m']['d'] = adj_dict['m']['d'].to_sparse()

        adj_dict['a']['m'] = adj_dict['a']['m'].to_sparse()
        adj_dict['u']['m'] = adj_dict['u']['m'].to_sparse()
        adj_dict['d']['m'] = adj_dict['d']['m'].to_sparse()

    return label, ft_dict, adj_dict


def load_acm_4025(args, train_percent):
    data_path = '../data/acm/acm4025.pkl'
    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)

        p_label = label['p'][0]

        path='../data/acm/'
        file_name = path+'p_label_im_'+str(args.im_ratio)+'.pkl'
        if os.path.exists(file_name):
            with open(file_name, "rb+") as f:
                label = pickle.load(f)
        else:
            idx_train_p, idx_val_p, idx_test_p = train_val_test_split_imbalance(args,p_label.shape[0], train_percent, p_label, args.im_ratio)
            label = {}
            label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]
            with open(file_name, "wb") as f:
                pickle.dump(label, f)



        args.major_num = int(p_label.shape[0] * train_percent / np.unique(p_label).shape[0] /10)*10
        args.minor_num = int(args.major_num * args.im_ratio)


        adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
        adj_dict['p']['l'] = adj_dict['p']['l'].to_sparse()

        adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
        adj_dict['l']['p'] = adj_dict['l']['p'].to_sparse()

    return label, ft_dict, adj_dict


def load_dblp_4057(args, train_percent):
    data_path = '../data/dblp/dblp4057.pkl'
    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)
        labels = {}
        for key in label.keys():
            key_label = label[key][0]

            if key == args.target_type:
                path='../data/dblp/'
                file_name = path+'a_label_im_'+str(args.im_ratio)+'.pkl'
                if os.path.exists(file_name):
                    with open(file_name, "rb+") as f:
                        label_a = pickle.load(f)
                else:
                    idx_train_a, idx_val_a, idx_test_a = train_val_test_split_imbalance(args,key_label.shape[0], train_percent, key_label, args.im_ratio)
                    label_a = [key_label, idx_train_a, idx_val_a, idx_test_a]
                    with open(file_name, "wb") as f:
                        pickle.dump(label_a, f)
                labels[key] = label_a

                args.major_num = int(key_label.shape[0] * train_percent / np.unique(key_label).shape[0] /10)*10
                args.minor_num = int(args.major_num * args.im_ratio)


        adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
        adj_dict['p']['c'] = adj_dict['p']['c'].to_sparse()
        adj_dict['p']['t'] = adj_dict['p']['t'].to_sparse()

        adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
        adj_dict['c']['p'] = adj_dict['c']['p'].to_sparse()
        adj_dict['t']['p'] = adj_dict['t']['p'].to_sparse()

    return labels, ft_dict, adj_dict

def load_data(args, dataset,train_percent):
    if dataset=="mag":
        label, ft_dict, adj_dict=load_odbmag_4017(args,train_percent)
    if dataset=="dblp":
        label, ft_dict, adj_dict=load_dblp_4057(args,train_percent)
    if dataset=="acm":
        label, ft_dict, adj_dict=load_acm_4025(args,train_percent)
    if dataset=="imdb":
        label, ft_dict, adj_dict=load_imdb_3228(args,train_percent)

    return label, ft_dict, adj_dict


