import torch
import torch.nn as nn

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, AutoRecConfig

"""
[PreprocessConfig]
implicit            = False

[DataSplitConfig]
method              = 'holdout'

[DatasetConfig]
method              = 'matrix'
num_negative        = 0

[ModelConfig]
algorithm           = 'AutoRec
"""


class AutoRec(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=AutoRecConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step

        self.n_user = input['n_user']
        self.n_item = input['n_item']
        self.fill_na_as = config.fill_na_as

        # for encoder
        self.pre_encoder = nn.Linear(self.n_item, config.dim_latent)
        self.encoder = nn.Sigmoid()

        # for decoder
        self.pre_decoder = nn.Linear(config.dim_latent, self.n_item)
        self.decoder = nn.Identity()

        self._init_weight()
        self._set_optimizer(config_opt)
        self._set_scheduler(config_opt)
        self._set_loss_fn()
        self.eval_fn = self._get_eval_fn(config.evaluation)

    def _init_weight(self):
        self.pre_encoder.weight.data.normal_(0, 0.03)
        self.pre_decoder.weight.data.normal_(0, 0.03)

    def _set_loss_fn(self):
        self.measure_train['measure'] = "MSE"
        self.loss_fn = nn.MSELoss()

    def fit(self, train_loader, val_data):
        print(f'\n[AutoRec]'+' Train '.center(100, '='))
        print("[AutoRec] ".ljust(20, ' ') +\
              f"(train: {self.measure_train['measure']})".ljust(20, ' ') +\
              f"(validation: {self.measure_val['measure']})".ljust(20, ' '))

        for epoch in range(1, self.epochs+1):

            train_loss = 0
            train_cnt = 0

            for batch_idx, samples in enumerate(train_loader):
                self.train()

                _, ratings, implicit = samples
                pred = self.forward(r_by_items=ratings, mask=implicit)
                masked_pred = pred[pred!=0]
                masked_true = torch.mul(ratings, implicit)
                masked_true = masked_true[masked_true!=0]
                loss = self.backward(masked_pred, masked_true)

                train_loss += loss
                train_cnt += 1

            self.scheduler.step()

            performance = self.evaluate(train_loader, val_data)

            self.measure_train['values'].append((train_loss/train_cnt).detach().numpy().tolist())
            self.measure_val['values'].append(performance)

            if epoch % self.print_step == 0:
                print(f"(epoch {epoch}) ".rjust(20, ' ') +\
                      f"{train_loss/train_cnt:.4f}".ljust(20, ' ') + f"{performance:.4f}".ljust(20, ' '))

    def forward(self, r_by_items, mask):
        r_by_items[r_by_items == 0] = self.fill_na_as

        pre_encoded = self.pre_encoder(r_by_items)
        encoded = self.encoder(pre_encoded)
        pre_decoded = self.pre_decoder(encoded)
        decoded = self.decoder(pre_decoded)

        ret = torch.mul(decoded, mask)

        return ret

    def backward(self, pred, true):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, true)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, r_by_items):
        r_by_items[r_by_items == 0] = self.fill_na_as

        pre_encoded = self.pre_encoder(r_by_items)
        encoded = self.encoder(pre_encoded)
        pre_decoded = self.pre_decoder(encoded)
        decoded = self.decoder(pre_decoded)

        return decoded

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # gt
            gt = eval_data['rating'].values.tolist()

            # prediction
            ratings = train_loader.dataset.rating_matrix
            ratings = torch.FloatTensor(ratings)

            pred_matrix = self.predict(r_by_items=ratings)
            test_idx_users = eval_data['user_id'].values
            test_idx_items = eval_data['item_id'].values
            prediction = pred_matrix[test_idx_users, test_idx_items].tolist()

            performance = self.eval_fn(pred=prediction, gt=gt)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[AutoRec]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[AutoRec] (test: {self.measure_test["measure"]}) = {performance:.4f}')
