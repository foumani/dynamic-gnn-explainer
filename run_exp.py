import evolvegcn.utils as u
import torch
import torch.distributed as dist
import numpy as np
import random
import os

# datasets

import evolvegcn.explanation_dl as exp

import evolvegcn.node_cls_tasker as nct
import evolvegcn.splitter as sp

# models
import evolvegcn.models as mls
import evolvegcn.egcn_h as egcn_h
import evolvegcn.egcn_h_exp as egcn_h_exp
import evolvegcn.Explanation_Cross_Entropy as ece

import evolvegcn.trainer as tr


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower() == 'none':
        if type == 'int':
            return random.randrange(param_min, param_max + 1)
        elif type == 'logscale':
            interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval, 1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param


def build_random_hyper_params(args):
    args.learning_rate = random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max,
                                            type='logscale')
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

    args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max,
                                             type='int')

    args.gcn_parameters['feats_per_node'] = random_param_value(args.gcn_parameters['feats_per_node'],
                                                               args.gcn_parameters['feats_per_node_min'],
                                                               args.gcn_parameters['feats_per_node_max'], type='int')
    args.gcn_parameters['layer_1_feats'] = random_param_value(args.gcn_parameters['layer_1_feats'],
                                                              args.gcn_parameters['layer_1_feats_min'],
                                                              args.gcn_parameters['layer_1_feats_max'], type='int')
    if args.gcn_parameters['layer_2_feats_same_as_l1']: #or args.gcn_parameters[
    #     'layer_2_feats_same_as_l1'].lower() == 'true':
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
    else:
        args.gcn_parameters['layer_2_feats'] = random_param_value(args.gcn_parameters['layer_2_feats'],
                                                                  args.gcn_parameters['layer_1_feats_min'],
                                                                  args.gcn_parameters['layer_1_feats_max'], type='int')
    args.gcn_parameters['lstm_l1_feats'] = random_param_value(args.gcn_parameters['lstm_l1_feats'],
                                                              args.gcn_parameters['lstm_l1_feats_min'],
                                                              args.gcn_parameters['lstm_l1_feats_max'], type='int')
    if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters[
        'lstm_l2_feats_same_as_l1']:
        args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
    else:
        args.gcn_parameters['lstm_l2_feats'] = random_param_value(args.gcn_parameters['lstm_l2_feats'],
                                                                  args.gcn_parameters['lstm_l1_feats_min'],
                                                                  args.gcn_parameters['lstm_l1_feats_max'], type='int')
    args.gcn_parameters['cls_feats'] = random_param_value(args.gcn_parameters['cls_feats'],
                                                          args.gcn_parameters['cls_feats_min'],
                                                          args.gcn_parameters['cls_feats_max'], type='int')
    return args


def build_dataset(args):
    if args.data == 'explanation':
        return exp.Explanation_Dataset(args)
    else:
        raise NotImplementedError('only arxiv has been implemented')


def build_tasker(args, dataset):
    if args.task == 'node_cls':
        return nct.Node_Cls_Tasker(args, dataset)
    # elif args.task == 'static_node_cls':
    #     return nct.Static_Node_Cls_Tasker(args, dataset)

    else:
        raise NotImplementedError('still need to implement the other tasks')


def build_gcn(args, tasker):
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node

    if args.model == 'egcn_h':
        return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)
    elif args.model == 'egcn_h_exp':
        return egcn_h_exp.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)

    else:
        raise NotImplementedError('need to finish modifying the models')


def build_classifier(args, tasker):
    if 'node_cls' == args.task or 'static_node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    if 'gru' in args.model or 'lstm' in args.model:
        in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
    elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
        in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
    else:
        in_feats = args.gcn_parameters['layer_2_feats'] * mult

    return mls.Classifier(args, in_features=in_feats, out_features=tasker.num_classes).to(args.device)


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = u.create_parser()
    args = u.parse_args(parser)

    global rank, wsize, use_cuda
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device = 'cpu'
    if args.use_cuda:
        args.device = 'cuda'
    print("use CUDA:", args.use_cuda, "- device:", args.device)
    try:

        dist.init_process_group(backend='mpi')  # , world_size=4
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank)  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
                                                                                   wsize)))

    if args.seed is None and args.seed != 'None':
        seed = 123 + rank  # int(time.time())+rank
    else:
        seed = args.seed  # +rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed = seed
    args.rank = rank
    args.wsize = wsize

    # Assign the requested random hyper parameters
    args = build_random_hyper_params(args)

    # build the dataset
    dataset = build_dataset(args)
    # build the tasker
    tasker = build_tasker(args, dataset)
    # build the splitter
    splitter = sp.splitter(args, tasker)
    # build the models
    gcn = build_gcn(args, tasker)
    classifier = build_classifier(args, tasker)
    # build a loss

    cross_entropy = ece.Cross_Entropy(args, dataset).to(args.device)

    # trainer
    trainer = tr.Trainer(args,
                         splitter=splitter,
                         gcn=gcn,
                         classifier=classifier,
                         comp_loss=cross_entropy,
                         dataset=dataset,
                         num_classes=tasker.num_classes)

    trainer.train()
