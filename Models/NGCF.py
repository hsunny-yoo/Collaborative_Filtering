import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, NGCFConfig

"""
[PreprocessConfig]
implicit           = False -> True

[DataSplitConfig]
method             = holdout

[DatasetConfig]
metohd             = pairwise
num_negative       > 0
between_observed   = False
"""


class NGCF(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=NGCFConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step

        self.n_user = input['n_user']
        self.n_item = input['n_item']
        self.k = config.k
        self.num_layer = config.num_layer
        self.dropout_msg = config.dropout_msg
        self.dropout_node = config.dropout_node

        self.user_embedding = nn.Embedding(self.n_user, config.dim_embedding)
        self.item_embedding = nn.Embedding(self.n_item, config.dim_embedding)
        self.embeddings = nn.ParameterDict()
        for k in range(1, config.num_layer+1):
            self.embeddings.update(
                {f'W1_{k}': nn.Parameter(nn.init.normal_(torch.empty(config.dim_embedding, config.dim_embedding))),
                 f'b1_{k}': nn.Parameter(nn.init.normal_(torch.empty(1, config.dim_embedding))),
                 f'W2_{k}': nn.Parameter(nn.init.normal_(torch.empty(config.dim_embedding, config.dim_embedding))),
                 f'b2_{k}': nn.Parameter(nn.init.normal_(torch.empty(1, config.dim_embedding)))})
        self.dropout = nn.Dropout(self.dropout_msg)

        self._init_weight()
        self._set_optimizer(config_opt)
        self._set_scheduler(config_opt)
        self._set_loss_fn()
        self.eval_fn = self._get_eval_fn(config.evaluation)

    def _init_weight(self):
        self.user_embedding.weight.data.normal_(0, 0.01)
        self.item_embedding.weight.data.normal_(0, 0.01)

    def _set_loss_fn(self):
        self.measure_train['measure'] = "BPRloss"
        self.loss_fn = lambda x: -torch.log(torch.sigmoid(x)).sum()

    def fit(self, train_loader, val_data):
        print(f'\n[NGCF]'+' Train '.center(100, '='))
        print("[NGCF] ".ljust(20, ' ') +\
              f"(train: {self.measure_train['measure']})".ljust(20, ' ') +\
              f"(validation: {self.measure_val['measure']})".ljust(20, ' '))

        self.laplacian = self._get_Laplacian(train_loader.dataset.implicit_matrix)

        for epoch in range(1, self.epochs+1):

            train_loss = 0
            train_cnt = 0

            for batch_idx, samples in enumerate(train_loader):
                self.train()

                users, items_pos, items_neg = samples
                pred = self.forward(users=users, items_pos=items_pos, items_neg=items_neg)
                loss = self.backward(pred)

                train_loss += loss
                train_cnt += 1

            self.scheduler.step()

            performance = self.evaluate(train_loader, val_data)

            self.measure_train['values'].append((train_loss/train_cnt).detach().numpy().tolist())
            self.measure_val['values'].append(performance)

            if epoch % self.print_step == 0:
                print(f"(epoch {epoch}) ".rjust(20, ' ') +\
                      f"{train_loss/train_cnt:.4f}".ljust(20, ' ') + f"{performance:.4f}".ljust(20, ' '))

    def forward(self, users, items_pos, items_neg):
        user_emb, item_emb = self._get_ngcf_embedding()

        embedded_user = F.embedding(users, user_emb)
        embedded_pos = F.embedding(items_pos, item_emb)
        embedded_neg = F.embedding(items_neg, item_emb)

        prefer_pos = torch.mul(embedded_user, embedded_pos).sum(dim=1)
        prefer_neg = torch.mul(embedded_user, embedded_neg).sum(dim=1)

        ret = prefer_pos - prefer_neg

        return ret

    def backward(self, pred):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, users, items):
        user_emb, item_emb = self._get_ngcf_embedding()

        embedded_users = F.embedding(users, user_emb)
        embedded_items = F.embedding(items, item_emb)

        prefer = torch.mul(embedded_users, embedded_items).sum(dim=1)

        return prefer

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # prediction
            concat_emb_users, concat_emb_items = self._get_ngcf_embedding()
            prediction = torch.matmul(concat_emb_users, concat_emb_items.T)
            implicit = train_loader.dataset.implicit_matrix
            implicit_idx = np.where(implicit>0)
            prediction[implicit_idx] = 0.0
            prediction = [np.argsort(prediction[u].detach().numpy())[-1::-1].tolist() for u in range(self.n_user)]

            # testset
            test = [eval_data[eval_data['user_id']==u].sort_values('rating')['item_id'].values[-1::-1].tolist()\
                    for u in range(self.n_user)]

            performance = self.eval_fn(reco=prediction, gt=test, k=self.k)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[NGCF]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[NGCF] (test: {self.measure_test["measure"]}) = {performance:.4f}')

    def _get_Laplacian(self, rating_matrix):
        adjacency = np.zeros((self.n_user+self.n_item, self.n_user+self.n_item))
        adjacency[:self.n_user, self.n_user:] = rating_matrix
        adjacency[self.n_user:, :self.n_user] = rating_matrix.T

        num_neighbor = adjacency.sum(axis=1)
        sqrt_d = np.diag(1/np.sqrt(num_neighbor))

        ret = np.dot(sqrt_d, adjacency)
        ret = np.dot(ret, sqrt_d)

        return torch.FloatTensor(ret)

    def _get_ngcf_embedding(self):
        embedded_users = self.user_embedding.weight
        embedded_item = self.item_embedding.weight
        concat_emb = torch.cat([embedded_users, embedded_item])

        if self.dropout_node > 0:
            # @TODO
            pass

        embeddings = [concat_emb]
        for k in range(1, self.num_layer+1):

            lap_for_self = self.laplacian + torch.eye(self.n_user+self.n_item)
            msg_self = torch.matmul(lap_for_self, concat_emb)
            msg_self = torch.matmul(msg_self, self.embeddings[f'W1_{k}']) + self.embeddings[f'b1_{k}']

            squared_emb = torch.mul(concat_emb, concat_emb)
            sum_emb = torch.matmul(self.laplacian, squared_emb)
            msg_neighbor = torch.matmul(sum_emb, self.embeddings[f'W2_{k}']) + self.embeddings[f'b2_{k}']

            activated = F.leaky_relu(msg_self + msg_neighbor, 0.2)

            dropped = self.dropout(activated)
            concat_emb = dropped

            normed = F.normalize(dropped, p=2, dim=1)

            embeddings.append(normed)

        embeddings = torch.concat(embeddings, 1)

        users = embeddings[:self.n_user, :]
        items = embeddings[self.n_user:, :]

        return users, items