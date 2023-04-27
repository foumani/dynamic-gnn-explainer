import evolvegcn.utils as u
import os

import tarfile
import numpy as np
import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class Explanation_Dataset():
    def __init__(self, args):
        args.explanation_args = u.Namespace(args.explanation_args)

        # self.course = args.explanation_args.course
        self.num_classes = args.explanation_args.num_classes
        #
        # phase = 'test_phase' if args.test_phase else 'train_phase'
        #
        # args.explanation_args.raw_data_folder = args.explanation_args.folder + '/baseline/' + phase + '/' + \
        #                                          self.course + '_' + str(self.num_classes - 1)
        #
        # args.explanation_args.folder = args.explanation_args.folder + '/' + phase + '/' + \
        #                                 self.course + '_' + str(self.num_classes - 1)

        self.train_edges = self.load_edges(args, dataset_type='train')
        self.test_edges = self.load_edges(args, dataset_type='test')

        self.train_nodes_labels = self.load_node_labels(args, dataset_type='train')
        self.test_nodes_labels = self.load_node_labels(args, dataset_type='test')

        self.train_nodes, self.train_nodes_feats = self.load_node_feats(args, dataset_type='train')
        self.test_nodes, self.test_nodes_feats = self.load_node_feats(args, dataset_type='test')

        # self.train_behaviours = self.load_behaviour_data(args, dataset_type='train')
        # self.test_behaviours = self.load_behaviour_data(args, dataset_type='test')

        if not args.test_phase:
            self.dev_edges = self.load_edges(args, dataset_type='dev')
            self.dev_nodes_labels = self.load_node_labels(args, dataset_type='dev')
            self.dev_nodes, self.dev_nodes_feats = self.load_node_feats(args, dataset_type='dev')
            # self.dev_behaviours = self.load_behaviour_data(args, dataset_type='dev')
        else:
            self.dev_edges = {}
            self.dev_nodes_labels = torch.empty(0, 3)
            self.dev_nodes = torch.empty(0, 3)
            self.dev_nodes_feats = torch.empty(0, 3)
            # self.dev_behaviours = {}

    def load_behaviour_data(self, args, dataset_type):
        df = pd.read_csv(f'{args.explanation_args.raw_data_folder}/{dataset_type}_dataset.csv')
        # df.insert(0, 'user_id', np.arange(20, 20 + len(df)))

        behvaiours = {}
        for i in range(args.num_hist_steps):
            behaviour_nodes = [[0] * 20 for _ in range(20)]
            user_clicks = [[float(value) for _, value in row.items()] for _, row in
                           df.iloc[:, i * 20:(i + 1) * 20].iterrows()]
            behvaiours[i] = torch.DoubleTensor((behaviour_nodes + user_clicks)).float()

        return behvaiours

    def load_node_labels(self, args, dataset_type):
        df = pd.read_csv(f'{args.explanation_args.folder}/node_labels_{dataset_type}.csv')
        if dataset_type == 'train':
            y_true = df['1']
            weights = {0:0.2, 1:0.05, 2:0.75}
            class_weights = compute_class_weight(class_weight=weights,
                                                 classes=np.unique(y_true), y=np.array(y_true))
            print(class_weights)
            args.class_weights = class_weights
        labels = [[float(value) for _, value in row.items()] for _, row in df.iterrows()]
        labels = torch.DoubleTensor(labels).long()
        # lcols = u.Namespace({'nid': 0,
        #                      'label': 1})

        return labels

    def load_node_feats(self, args, dataset_type):
        df = pd.read_csv(f'{args.explanation_args.folder}/node_feat_{dataset_type}.csv')
        nodes = [[float(value) for _, value in row.items()] for _, row in df.iterrows()]
        nodes = torch.DoubleTensor(nodes)

        nodes_feats = nodes[:, 1:]

        if dataset_type == 'train':
            self.train_num_nodes = len(nodes)
        elif dataset_type == 'dev':
            self.dev_num_nodes = len(nodes)
        else:
            self.test_num_nodes = len(nodes)

        self.feats_per_node = nodes.size(1) - 1

        return nodes, nodes_feats.float()

    def load_edges(self, args, dataset_type):
        edges = []
        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'weight': 2,
                            'time': 3})
        for item in range(int(args.num_hist_steps)):
            df = pd.read_csv(f'{args.explanation_args.folder}/el_{dataset_type}_week_{str(item)}.csv')
            data = [[float(value) for _, value in row.items()] for _, row in df.iterrows()]
            data = torch.DoubleTensor(data)

            time_col = torch.zeros(data.size(0), 1, dtype=torch.long) + item

            data = torch.cat([data, time_col], dim=1)

            data = torch.cat([data, data[:, [cols.target,
                                             cols.source,
                                             cols.weight,
                                             cols.time]]])

            edges.append(data)

        edges = torch.cat(edges)
        # new_edges = edges.long()
        # _, new_edges[:, [cols.source, cols.target, cols.weight]] = edges[:,
        #                                                            [cols.source, cols.target, cols.weight]].unique(
        # return_inverse=True)

        # time aggregation
        edges[:, cols.time] = u.aggregate_by_time(edges[:, cols.time], args.explanation_args.aggr_time)

        # self.num_nodes = int(edges[:, [cols.source, cols.target]].max() + 1)

        # ids = edges[:, cols.source] * self.num_nodes + edges[:, cols.target]
        # self.num_non_existing = float(self.num_nodes ** 2 - ids.unique().size(0))

        # self.max_time = edges[:, cols.time].max()
        # self.min_time = edges[:, cols.time].min()

        idx = edges[:, [cols.target,
                        cols.source,
                        cols.time]].long()

        vals = edges[:, cols.weight]

        return {'idx': idx, 'vals': vals}
