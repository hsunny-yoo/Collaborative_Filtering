import torch
import torch.nn as nn
import numpy as np

from Models.BaseModel import BaseModel
from Models.config import OptimizerConfig, CDAEConfig

"""
[PreprocessConfig]
implcit             = True
implicit_threshold  = 4

[DatasetConfig]
method              = 'matrix'
num_negative        > 0
between_observed    = False

[ModelConfig]
algorithm           = 'CDAE'
"""


class CDAE(BaseModel):
    def __init__(self, input, config_train, config_opt=OptimizerConfig, config=CDAEConfig):
        super().__init__()

        self.epochs = config_train.epochs
        self.print_step = config_train.print_step
        self.k = config.k

        self.n_user = input['n_user']
        self.n_item = input['n_item']
        self.corrupt_ratio = config.corrupt_ratio

        # for encoder
        self.pre_encoder = nn.Linear(self.n_item, config.dim_latent)
        self.encoder = nn.Sigmoid()

        # for user-bias
        self.user_embedding = nn.Embedding(self.n_user, config.dim_latent)

        # for decoder
        self.pre_decoder = nn.Linear(config.dim_latent, self.n_item)
        self.decoder = nn.Sigmoid()

        self._init_weight()
        self._set_optimizer(config_opt)
        self._set_scheduler(config_opt)
        self._set_loss_fn()
        self.eval_fn = self._get_eval_fn(config.evaluation)

    def _init_weight(self):
        self.pre_encoder.weight.data.normal_(0, 0.03)
        self.pre_decoder.weight.data.normal_(0, 0.03)

    def _set_loss_fn(self):
        self.measure_train['measure'] = "BCE"
        self.loss_fn = nn.BCELoss()

    def fit(self, train_loader, val_data):
        print(f'\n[CDAE]'+' Train '.center(100, '='))
        print("[CDAE] ".ljust(20, ' ') +\
              f"(train: {self.measure_train['measure']})".ljust(20, ' ') +\
              f"(validation: {self.measure_val['measure']})".ljust(20, ' '))

        for epoch in range(1, self.epochs+1):

            train_loss = 0
            train_cnt = 0

            for batch_idx, samples in enumerate(train_loader):
                self.train()

                users, ratings, implicit = samples

                # for decoder update
                pred_encoder, pred_decoder, encoder_mask, decoder_mask = self.forward(users=users,
                                                                                      r_by_items=ratings,
                                                                                      mask=implicit)
                loss_decoder = self.backward(pred_decoder, torch.mul(ratings, decoder_mask), 'decoder')

                # for encoder, bias update
                pred_encoder, pred_decoder, encoder_mask, decoder_mask = self.forward(users=users,
                                                                                      r_by_items=ratings,
                                                                                      mask=implicit,
                                                                                      encoder_mask=encoder_mask,
                                                                                      decoder_mask=decoder_mask)
                loss_encoder = self.backward(pred_encoder, torch.mul(ratings, encoder_mask), 'encoder')
                train_loss += (loss_decoder+loss_encoder)/2
                train_cnt += 1

            self.scheduler.step()

            performance = self.evaluate(train_loader, val_data)

            self.measure_train['values'].append((train_loss/train_cnt).detach().numpy().tolist())
            self.measure_val['values'].append(performance)

            if epoch % self.print_step == 0:
                print(f"(epoch {epoch}) ".rjust(20, ' ') +\
                      f"{train_loss/train_cnt:.4f}".ljust(20, ' ') + f"{performance:.4f}".ljust(20, ' '))

    def forward(self, users, r_by_items, mask, encoder_mask=None, decoder_mask=None):
        if encoder_mask is None and decoder_mask is None:
            pos_mask = (r_by_items!=0.0).type(torch.int64)
            corrupted = self._denoise(pos_mask)

            encoder_mask = torch.tensor((corrupted>0.0).astype(np.int64))
            decoder_mask = mask
        else:
            corrupted = torch.mul(r_by_items, encoder_mask/(1-self.corrupt_ratio))

        corrupted = torch.FloatTensor(corrupted)

        pre_encoded = self.pre_encoder(corrupted)
        encoded = self.encoder(pre_encoded)
        embedded_user = self.user_embedding(users)
        pre_decoded = self.pre_decoder(encoded+embedded_user)
        decoded = self.decoder(pre_decoded)

        encoder_masked = torch.mul(encoder_mask, decoded)
        decoder_masked = torch.mul(decoder_mask, decoded)

        return encoder_masked, decoder_masked, encoder_mask, decoder_mask

    def backward(self, pred, true, obj):
        if obj == 'decoder':
            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, true)
            self.pre_encoder.weight.requires_grad = False
            self.user_embedding.weight.requires_grad = False
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.pre_encoder.weight.requires_grad = True
            self.user_embedding.weight.requires_grad = True

        elif obj == 'encoder':
            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, true)
            self.pre_decoder.weight.requires_grad = False
            loss.backward()
            self.optimizer.step()
            self.pre_decoder.weight.requires_grad = True

        return loss

    def predict(self, users, r_by_items):
        pre_encoded = self.pre_encoder(r_by_items)
        encoded = self.encoder(pre_encoded)
        embedded_user = self.user_embedding(users)
        pre_decoded = self.pre_decoder(encoded+embedded_user)
        decoded = self.decoder(pre_decoded)

        return decoded

    def evaluate(self, train_loader, eval_data):
        with torch.no_grad():
            self.eval()

            # gt
            gt = [eval_data[eval_data['user_id']==u].sort_values('rating')['item_id'].values[-1::-1].tolist() \
                  for u in range(self.n_user)]

            # prediction
            users= range(self.n_user)
            users = torch.LongTensor(users)
            ratings = train_loader.dataset.rating_matrix
            ratings = torch.FloatTensor(ratings)

            pred_matrix = self.predict(users=users, r_by_items=ratings)
            implicit = train_loader.dataset.implicit_matrix
            implicit_idx = np.where(implicit>0)
            pred_matrix[implicit_idx] = 0.0
            prediction = [np.argsort(pred_matrix[u].detach().numpy())[-1::-1].tolist() for u in range(self.n_user)]

            performance = self.eval_fn(reco=prediction, gt=gt, k=self.k)

        return performance

    def test(self, train_loader, test_data):
        print(f'\n[CDAE]'+' Evaluate '.center(100, '='))

        performance = self.evaluate(train_loader, test_data)
        self.measure_test['values'].append(performance)

        print(f'[CDAE] (test: {self.measure_test["measure"]}) = {performance:.4f}')

    def _denoise(self, pos_mask):
        ret = np.zeros(pos_mask.shape)
        row_idx, col_idx = np.where(pos_mask!=0)
        num_pos_element = row_idx.shape[0]

        probs = [self.corrupt_ratio, 1-self.corrupt_ratio]
        corrupted_rating = np.random.choice([0, 1/(1-self.corrupt_ratio)], num_pos_element, p=probs)

        ret[row_idx, col_idx] = corrupted_rating

        return ret
