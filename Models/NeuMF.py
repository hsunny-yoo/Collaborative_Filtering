import torch
import torch.nn as nn
import numpy as np

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, NeuMFConfig

"""
[PreprocessConfig]
implicit            = True

[DataSplitConfig]
method              = 'leave_k_out'
leave_k             = 1

[DatasetConfig]
method              = 'pointwise'
num_negative        > 0
between_observed    = False

[ModelConfig]
algorithm           = 'GMF'
"""


class NeuMF(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=NeuMFConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step

        self.n_user = input['n_user']
        self.n_item = input['n_item']
        self.k = config.k
        self.sample = config.sample
        self.alpha = config.alpha

        # for GMF
        self.user_embedding_gmf = nn.Embedding(self.n_user, config.dim_latent_gmf)
        self.item_embedding_gmf = nn.Embedding(self.n_item, config.dim_latent_gmf)
        self.sum_weight_gmf = nn.Linear(config.dim_latent_gmf, 1, bias=False)

        # for MLP
        self.user_embedding_mlp = nn.Embedding(self.n_user, config.dim_latent_mlp)
        self.item_embedding_mlp = nn.Embedding(self.n_item, config.dim_latent_mlp)
        self.fc1 = nn.Linear(config.dim_latent_mlp*2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.sum_weight_mlp = nn.Linear(8, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self._init_weight()
        self._set_optimizer(config_opt)
        self._set_scheduler(config_opt)
        self._set_loss_fn()
        self.eval_fn = self._get_eval_fn(config.evaluation)

    def _init_weight(self):
        self.user_embedding_gmf.weight.data.normal_(0, 0.01)
        self.item_embedding_gmf.weight.data.normal_(0, 0.01)
        self.user_embedding_mlp.weight.data.normal_(0, 0.01)
        self.item_embedding_mlp.weight.data.normal_(0, 0.01)

    def _set_loss_fn(self):
        self.measure_train['measure'] = "BCE"
        self.loss_fn = nn.BCELoss()

    def fit(self, train_loader, val_data):
        print(f'\n[NeuFM]'+' Train '.center(100, '='))
        print("[NeuFM] ".ljust(20, ' ') +\
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
        # GMF
        embedded_users_gmf = self.user_embedding_gmf(users)
        embedded_items_gmf = self.item_embedding_gmf(items)
        interact = torch.mul(embedded_users_gmf, embedded_items_gmf)
        weighted_sum_gmf = self.sum_weight_gmf(interact)

        # MLP
        embedded_users_mlp = self.user_embedding_mlp(users)
        embedded_items_mlp = self.item_embedding_mlp(items)
        concated = torch.cat([embedded_users_mlp, embedded_items_mlp], dim=-1)
        z1 = self.fc1(concated)
        o1 = self.relu(z1)
        z2 = self.fc2(o1)
        o2 = self.relu(z2)
        z3 = self.fc3(o2)
        o3 = self.relu(z3)
        weighted_sum_mlp = self.sum_weight_mlp(o3)

        # concat
        weighted_sum = self.alpha*weighted_sum_gmf + (1-self.alpha)*weighted_sum_mlp
        ret = self.sigmoid(weighted_sum)

        return ret

    def backward(self, pred, true):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, true)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, users, items):
        # GMF
        embedded_users_gmf = self.user_embedding_gmf(users)
        embedded_items_gmf = self.item_embedding_gmf(items)
        interact = torch.mul(embedded_users_gmf, embedded_items_gmf)
        weighted_sum_gmf = self.sum_weight_gmf(interact)

        # MLP
        embedded_users_mlp = self.user_embedding_mlp(users)
        embedded_items_mlp = self.item_embedding_mlp(items)
        concated = torch.cat([embedded_users_mlp, embedded_items_mlp], dim=-1)
        z1 = self.fc1(concated)
        o1 = self.relu(z1)
        z2 = self.fc2(o1)
        o2 = self.relu(z2)
        z3 = self.fc3(o2)
        o3 = self.relu(z3)
        weighted_sum_mlp = self.sum_weight_mlp(o3)

        # concat
        weighted_sum = self.alpha*weighted_sum_gmf + (1-self.alpha)*weighted_sum_mlp
        ret = self.sigmoid(weighted_sum)

        return ret

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # prediction
            implicit = train_loader.dataset.implicit_matrix
            neg_prob = (implicit==0)
            neg_prob[eval_data['user_id'].values, eval_data['item_id'].values] = False
            neg_prob = neg_prob.astype(np.float64)
            neg_prob /= neg_prob.sum(axis=1)[:, np.newaxis]
            neg_sample = [np.random.choice(self.n_item, self.sample, replace=False, p=prob).tolist()\
                        for prob in neg_prob]
            neg_sample = [neg_sample[i]+eval_data[eval_data['user_id']==i]['item_id'].to_list() for i in range(self.n_user)]
            neg_sample = np.array(neg_sample)
            num_sample = neg_sample.shape[1]
            neg_item_idx = neg_sample.reshape(-1)
            neg_user_idx = np.array([[u]*num_sample for u in range(self.n_user)]).reshape(-1)

            users = torch.LongTensor(neg_user_idx)
            items = torch.LongTensor(neg_item_idx)
            pred_mat = self.predict(users=users, items=items)
            pred_mat = pred_mat.reshape(self.n_user, -1)
            prediction_idx = [np.argsort(pred_mat[u].detach().numpy())[-1::-1].tolist() for u in range(self.n_user)]
            neg_item_idx = neg_item_idx.reshape(self.n_user, -1)
            prediction = [list(neg_item_idx[i][prediction_idx[i]]) for i in range(self.n_user)]

            # gt
            gt = [eval_data[eval_data['user_id'] == u].sort_values('rating')['item_id'].values[-1::-1].tolist() \
                  for u in range(self.n_user)]

            performance = self.eval_fn(reco=prediction, gt=gt, k=self.k)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[NeuMF]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[NeuMF] (test: {self.measure_test["measure"]}) = {performance:.4f}')
