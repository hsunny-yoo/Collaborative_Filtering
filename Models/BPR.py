import torch
import torch.nn as nn
import numpy as np

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, BPRConfig

"""
[PreprocessConfig]
implicit                = True

[DataSplitConfig]
method                  = 'leave_k_out'

[DatasetConfig]
method                  = 'pairwise'
num_negative            > 0
between_observed        = False

[ModelConfig]
algorithm               = 'BPR'
"""


class BPR(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=BPRConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step

        self.n_user = input['n_user']
        self.n_item = input['n_item']

        self.user_embedding = nn.Embedding(self.n_user, config.dim_latent)
        self.item_embedding = nn.Embedding(self.n_item, config.dim_latent)

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
        print(f'\n[BPR]'+' Train '.center(100, '='))
        print("[BPR] ".ljust(20, ' ') +\
              f"(train: {self.measure_train['measure']})".ljust(20, ' ') +\
              f"(validation: {self.measure_val['measure']})".ljust(20, ' '))

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
        embedded_user = self.user_embedding(users)
        embedded_pos = self.item_embedding(items_pos)
        embedded_neg = self.item_embedding(items_neg)

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
        embedded_user = self.user_embedding(users)
        embedded_item = self.item_embedding(items)

        prefer = torch.mul(embedded_user, embedded_item).sum(dim=1)

        return prefer

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # prediction
            emb_users = self.user_embedding.weight.data
            emb_items = self.item_embedding.weight.data
            pred_mat = torch.matmul(emb_users, emb_items.T)
            implicit = train_loader.dataset.implicit_matrix
            implicit_idx = np.where(implicit>0)
            pred_mat[implicit_idx] = 0.0
            prediction = [np.argsort(pred_mat[u].detach().numpy())[-1::-1].tolist() for u in range(self.n_user)]

            # gt
            test = [eval_data[eval_data['user_id']==u].sort_values('rating')['item_id'].values[-1::-1].tolist()\
                    for u in range(self.n_user)]

            performance = self.eval_fn(reco=prediction, gt=test)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[BPR]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[BPR] (test: {self.measure_test["measure"]}) = {performance:.4f}')
