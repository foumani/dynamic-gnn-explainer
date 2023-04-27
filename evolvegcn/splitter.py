from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import evolvegcn.utils as u

class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args, tasker):

        assert args.train_proportion + args.dev_proportion < 1, \
            'there\'s no space for test samples'
        #only the training one requires special handling on start, the others are fine with the split IDX.
        # start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
        # end = args.train_proportion
        #
        # end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(tasker,dataset_type='train')
        train = DataLoader(train, **args.data_loading_params)

        # start = end
        # end = args.dev_proportion + args.train_proportion
        # end = int(np.floor(tasker.data.max_time.type(torch.float) * end))

        dev = data_split(tasker, dataset_type='dev',test_phase=args.test_phase)

        dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])

        # start = end
        #
        # #the +1 is because I assume that max_time exists in the dataset
        # end = int(tasker.max_time) + 1

        test = data_split(tasker,dataset_type='test')

        test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])

        print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))

        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test



class data_split(Dataset):
    def __init__(self, tasker, dataset_type, test_phase=False, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.dataset_type = dataset_type
        self.kwargs = kwargs
        self.test_phase = test_phase

    def __len__(self):
        if (self.dataset_type == 'dev') and (self.test_phase):
            return 0
        return 1

    def __getitem__(self, item):
        t = self.tasker.get_sample(dataset_type=self.dataset_type)
        return t


