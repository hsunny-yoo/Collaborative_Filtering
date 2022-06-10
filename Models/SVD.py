import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, SVDConfig

"""
[DatasetConfig]
method          = 'pointwise'

[ModelConfig]
algorithm       = 'SVD'
"""


class SVD(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=SVDConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step

        self.mu = input['mu']
        self.user_embedding = nn.Embedding(input['n_user'], config.dim_latent)
        self.item_embedding = nn.Embedding(input['n_item'], config.dim_latent)
        self.user_bias = nn.Embedding(input['n_user'], 1)
        self.item_bias = nn.Embedding(input['n_item'], 1)

        self._init_weight()
        self._set_optimizer(config_opt)
        self._set_scheduler(config_opt)
        self._set_loss_fn()
        self.eval_fn = self._get_eval_fn(config.evaluation)

    def _init_weight(self):
        self.user_embedding.weight.data.normal_(0, 1)
        self.item_embedding.weight.data.normal_(0, 1)
        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

    def _set_loss_fn(self):
        self.measure_train['measure'] = "MSE"
        self.loss_fn = F.mse_loss

    def fit(self, train_loader, val_data):
        print(f'\n[SVD]'+' Train '.center(100, '='))
        print("[SVD] ".ljust(20, ' ') +\
              f"(train: {self.measure_train['measure']})".ljust(20, ' ') +\
              f"(validation: {self.measure_val['measure']})".ljust(20, ' '))

        for epoch in range(1, self.epochs+1):

            train_loss = 0
            train_cnt = 0

            for batch_idx, samples in enumerate(train_loader):
                self.train()

                users, items, ratings, _ = samples
                pred = self.forward(users=users, items=items)
                loss = self.backward(pred, ratings)

                train_loss += loss
                train_cnt += 1

            self.scheduler.step()

            performance = self.evaluate(train_loader, val_data)

            self.measure_train['values'].append((train_loss/train_cnt).detach().numpy().tolist())
            self.measure_val['values'].append(performance)

            if epoch % self.print_step == 0:
                print(f"(epoch {epoch}) ".rjust(20, ' ') +\
                      f"{train_loss/train_cnt:.4f}".ljust(20, ' ') + f"{performance:.4f}".ljust(20, ' '))

    def forward(self, users, items):
        embedded_users = self.user_embedding(users)
        embedded_items = self.item_embedding(items)
        interact = torch.mul(embedded_users, embedded_items).sum(dim=1)
        bias_users = self.user_bias(users)
        bias_items = self.item_bias(items)

        return self.mu + bias_users + bias_items + interact.unsqueeze(-1)

    def backward(self, pred, true):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, true)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, users, items):
        embedded_users = self.user_embedding(users)
        embedded_items = self.item_embedding(items)
        interact = torch.mul(embedded_users, embedded_items).sum(dim=1)
        bias_users = self.user_bias(users)
        bias_items = self.item_bias(items)

        return self.mu + bias_users + bias_items + interact.unsqueeze(-1)

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # gt
            gt = eval_data['rating'].values.tolist()

            # prediction
            emb_users = self.user_embedding.weight.data
            emb_items = self.item_embedding.weight.data
            pred_mat = torch.matmul(emb_users, emb_items.T)
            test_idx_users = eval_data['user_id'].values
            test_idx_items = eval_data['item_id'].values
            prediction = pred_mat[test_idx_users, test_idx_items].tolist()

            performance = self.eval_fn(pred=prediction, gt=gt)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[SVD]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[SVD] (test: {self.measure_test["measure"]}) = {performance:.4f}')
