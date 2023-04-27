import torch
import evolvegcn.utils as u
import evolvegcn.explanation_logger as elogger
import time
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer


class Trainer:
    def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        self.args = args
        self.gcn = gcn
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.classifier = classifier
        self.comp_loss = comp_loss

        self.data = dataset
        self.num_classes = num_classes


        self.logger = elogger.Logger(args, self.num_classes)

        self.init_optimizers(args)
        self.cg_data = {}

    # if self.tasker.is_static:
    # 	adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
    # 	self.hist_adj_list = [adj_matrix]
    # 	self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

    def init_optimizers(self, args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
        # self.gcn_scheduler = StepLR(self.gcn_opt, gamma=0.1, step_size=10)
        if args.model == 'egcn_h':
            params = self.classifier.parameters()
            self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
        # self.classifier_scheduler = StepLR(self.classifier_opt, gamma=0.1, step_size=10)
        self.gcn_opt.zero_grad()
        if args.model == 'egcn_h':
            self.classifier_opt.zero_grad()

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def save_explanation_checkpoint(self, num_epochs=-1):
        """Save pytorch model checkpoint.

        Args:
            - model         : The PyTorch model to save.
            - optimizer     : The optimizer used to train the model.
            - args          : A dict of meta-data about the model.
            - num_epochs    : Number of training epochs.
            - isbest        : True if the model has the highest accuracy so far.
            - cg_dict       : A dictionary of the sampled computation graphs.
        """
        filename = f'checkpoints/checkpoint.pth.tar'
        torch.save(
            {
                "epoch": num_epochs,
                "gcn_dict": self.gcn.state_dict,
                "gcn_optimizer": self.gcn_opt.state_dict(),
                "cg":self.cg_data
                # "classifier_dict": self.classifier.state_dict(),
                # "classifier_optimizer": self.classifier_opt.state_dict(),
                # # "cg": cg_dict,
            },
            filename,
        )

    def load_checkpoint(self, filename, model):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            # self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            # self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
            return 0

    def train(self):
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        trainingEpoch_loss = []
        validationEpoch_loss = []
        testEpoch_loss = []
        for e in range(self.args.num_epochs):
            eval_train, train_nodes_embs, mean_loss = self.run_epoch(self.splitter.train, e, 'TRAIN', grad=True)
            trainingEpoch_loss.append(mean_loss.cpu())

            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                eval_valid, valid_node_embs, valid_mean_loss = self.run_epoch(self.splitter.dev, e, 'VALID', grad=False)
                validationEpoch_loss.append(valid_mean_loss.cpu())
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    self.save_explanation_checkpoint(e)
                    print('### w' + str(self.args.rank) + ') ep ' + str(e) + ' - Best valid measure:' + str(eval_valid))

                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print('### w' + str(self.args.rank) + ') ep ' + str(e) + ' - Early stop.')
                        break

            if len(self.splitter.test) > 0 and eval_valid == best_eval_valid and e > self.args.eval_after_epochs:
                eval_test, test_node_embs, test_mean_loss = self.run_epoch(self.splitter.test, e, 'TEST', grad=False)
                testEpoch_loss.append(test_mean_loss.cpu())


    def run_epoch(self, split, epoch, set_name, grad):
        t0 = time.time()
        log_interval = 999
        if set_name == 'TEST':
            log_interval = 1
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        for s in split:
            # if self.tasker.is_static:
            # 	s = self.prepare_static_sample(s)
            # else:
            s = self.prepare_sample(s, set_name)

            predictions, nodes_embs = self.predict(s.hist_adj_list,
                                                   s.hist_ndFeats_list,
                                                   s.label_sp['idx'],
                                                   s.node_mask_list, set_name,
                                                   s.raw_data_list)

            if set_name == 'TRAIN':
                self.cg_data['label'] = s.label_sp['vals']
            loss = self.comp_loss(predictions, s.label_sp['vals'])
            # print(loss)
            if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj=s.label_sp['idx'])
            elif set_name in ['TRAIN']:
                self.label_binarizer = LabelBinarizer().fit(s.label_sp['vals'].cpu().numpy())
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), plot_results=False)
            else:
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(),
                                          plot_results=(set_name == 'TEST'), label_binarizer=self.label_binarizer)
            if grad:
                self.optim_step(loss)
        # self.gcn_scheduler.step()
        # self.classifier_scheduler.step()
        torch.set_grad_enabled(True)
        eval_measure, mean_loss = self.logger.log_epoch_done(set_name)

        return eval_measure, nodes_embs, mean_loss

    def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, mask_list, set_name, raw_data_list=None):
        if self.args.model == 'egcn_h':
            nodes_embs_list = self.gcn(hist_adj_list, hist_ndFeats_list,
                                       mask_list)

            nodes_embs = nodes_embs_list[-1]
            predict_batch_size = 100000
            gather_predictions = []
            for i in range(1 + (node_indices.size(1) // predict_batch_size)):
                cls_input = self.gather_node_embs(nodes_embs,
                                                  node_indices[:, i * predict_batch_size:(i + 1) * predict_batch_size])
                predictions = self.classifier(cls_input)
                gather_predictions.append(predictions)
            gather_predictions = torch.cat(gather_predictions, dim=0)
        else:
            gather_predictions, nodes_embs_list = self.gcn(hist_adj_list, hist_ndFeats_list,
                                       mask_list, node_indices)

        if set_name == 'TRAIN':
            self.cg_data['adj'] = hist_adj_list
            self.cg_data['feat'] = hist_ndFeats_list
            self.cg_data['pred'] = gather_predictions


        return gather_predictions, nodes_embs_list

    def gather_node_embs(self, nodes_embs, node_indices):
        cls_input = []

        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input, dim=1)

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            if self.args.model == 'egcn_h':
                self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            if self.args.model == 'egcn_h':
                self.classifier_opt.zero_grad()

    def prepare_sample(self, sample, set_name):
        sample = u.Namespace(sample)

        if set_name == 'TRAIN':
            num_nodes = self.data.train_num_nodes

        elif set_name == 'VALID':
            num_nodes = self.data.dev_num_nodes

        else:
            num_nodes = self.data.test_num_nodes

        for i, adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj, torch_size=[num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            sample.node_mask_list[i] = node_mask.to(
                self.args.device).t()  # transposed to have same dimensions as scorer


        label_sp = self.ignore_batch_dim(sample.label_sp)

        label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    # def prepare_static_sample(self,sample):
    # 	sample = u.Namespace(sample)
    #
    # 	sample.hist_adj_list = self.hist_adj_list
    #
    # 	sample.hist_ndFeats_list = self.hist_ndFeats_list
    #
    # 	label_sp = {}
    # 	label_sp['idx'] =  [sample.idx]
    # 	label_sp['vals'] = sample.label
    # 	sample.label_sp = label_sp
    #
    # 	return sample

    def ignore_batch_dim(self, adj):
        adj['vals'] = adj['vals'][0]
        return adj

# def save_node_embs_csv(self, nodes_embs, indexes, file_name):
# 	csv_node_embs = []
# 	for node_id in indexes:
# 		orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])
#
# 		csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())
#
# 	pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
# 	#print ('Node embs saved in',file_name)
