from torch.utils.data import IterableDataset
import torch
import numpy as np


class PairwiseGenerator(IterableDataset):
    def __init__(self, train_data, n_user, n_item, config):
        super().__init__()

        self.data = train_data
        self.users = None
        self.items = None
        self.ratings = None
        self.num_data = None
        self.rating_matrix = None
        self.implicit_matrix = None

        self.num_negative = config.num_negative
        self.between_observed = config.between_observed
        self.num_user = n_user
        self.num_item = n_item

        self._set_data(train_data)

    def __len__(self):
        return self.num_data

    def __iter__(self):
        for i in range(self.num_data):
            user = torch.LongTensor(torch.tensor(self.users[i]))
            item_pos = torch.LongTensor(torch.tensor(self.items[i]))

            user_id = self.users[i]
            # implicit_items = torch.FloatTensor(self.implicit_matrix[user_id])

            if self.num_negative < 0:
                print('PairwiseGenerator can operate when the number of negative samples is over zero')
                print('Check config file')
                exit(0)

            else:
                pos_items = self.data[self.data['user_id']==user_id]['item_id'].unique()

                if self.between_observed:
                    neg_candidate = self.data[(self.data['user_id']==user_id)&
                                              (self.data['rating']<self.ratings[i])]['item_id'].values

                    num_sample = min(self.num_negative, neg_candidate.shape[0])
                    neg_items = np.random.choice(neg_candidate, num_sample, replace=False)

                else:
                    prob = np.ones(self.num_item)
                    prob[pos_items] = 0
                    prob /= prob.sum()

                    num_sample = min(self.num_negative, self.num_item-pos_items.shape[0])
                    neg_items = np.random.choice(self.num_item, num_sample, replace=False, p=prob)

                neg_items = neg_items.astype(np.int64)

                for neg in neg_items:
                    item_neg = torch.LongTensor(torch.tensor(neg))

                    yield user, item_pos, item_neg

    def _set_data(self, train_data):
        self.users = train_data['user_id'].values
        self.items = train_data['item_id'].values
        self.ratings = train_data['rating'].values

        self.num_data = self.users.shape[0]

        self.implicit_matrix = np.zeros((self.num_user, self.num_item))
        self.implicit_matrix[self.users,self.items] = 1

        self.rating_matrix = np.zeros((self.num_user, self.num_item))
        self.rating_matrix[self.users, self.items] = self.ratings