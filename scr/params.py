import argparse
import csv
import os
import errno
def mag_params(args):
    args.target_type="p"
    args.hidden_dim=[256,512]
    args.lr=0.0005
    args.l2_coef=5e-4
    args.meta_path=[]

    args.neighbor_scale=50
    args.temperature=2
    args.alpha= 0.05
    args.beta= 0.1

    return args


def imdb_params(args):
    args.target_type="m"
    args.hidden_dim=[256,512]
    args.lr=0.005
    args.l2_coef=5e-4
    args.meta_path=[["m","a","m","d"],["m","u","m","d"]]

    args.neighbor_scale=2
    args.temperature= 2
    args.alpha= 0.5
    args.beta= 0.01

    return args


def set_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mag") 
    parser.add_argument('--patience', type=int, default=100)       
    parser.add_argument('--log_dir', type=str, default="./result")
    parser.add_argument("--repeat_num", type=int, default=5)

    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--neighbor_scale", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--beta", type=int, default=1)

    parser.add_argument("--im_ratio", type=float, default=0.1)
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--target_type', type=str, default="p")
    parser.add_argument('--meta_path', type=str, default=[])
    parser.add_argument('--train_percent', type=float, default=0.06)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=[256,512])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)

    
    args, _ = parser.parse_known_args()
    if args.dataset == "mag":
        args = mag_params(args)
    if args.dataset == "imdb":
        args = imdb_params(args)
    args.log_dir = args.log_dir + '/' +args.dataset
    args.csv_file = args.log_dir +'/result_'+args.dataset +'_im_'+str(args.im_ratio)+'.csv'


    mkdir_p(args.log_dir)
    fieldnames = ['test_macro_f1', 'test_micro_f1', 'test_bacc']    
    with open(args.csv_file ,'w+',newline='')as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    return args



def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Save Results in Directory: {}".format(path))
        else:
            raise