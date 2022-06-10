from torch.utils.data import IterableDataset
import torch
import numpy as np


class MatrixGenerator(IterableDataset):
    def __init__(self, train_data, n_user, n_item, config):
        super().__init__()

        self.rating_matrix = None
        self.implicit_matrix = None

        self.num_data = n_user
        self.num_user = n_user
        self.num_item = n_item
        self.num_negative = config.num_negative

        self._set_data(train_data)

    def __len__(self):
        return self.num_data

    def __iter__(self):
        for i in range(self.num_data):
            user = torch.LongTensor(torch.tensor(i))
            rating_vector = torch.FloatTensor(self.rating_matrix[i])
            implicit_vector = torch.FloatTensor(self.implicit_matrix[i])

            if self.num_negative > 0:
                probs = np.ones(self.num_item)
                probs[implicit_vector.type(torch.int64)] = 0.0
                probs /= probs.sum()

                neg_idx = np.random.choice(self.num_item, self.num_negative, p=probs)
                implicit_vector[neg_idx] = 1

            yield user, rating_vector, implicit_vector


    def _set_data(self, train_data):
        users = train_data['user_id'].values
        items = train_data['item_id'].values
        ratings = train_data['rating'].values

        self.implicit_matrix = np.zeros((self.num_user, self.num_item))
        self.implicit_matrix[users,items] = 1

        self.rating_matrix = np.zeros((self.num_user, self.num_item))
        self.rating_matrix[users,items] = ratings