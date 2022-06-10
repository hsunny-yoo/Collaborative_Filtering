import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from Models import utils_evaluation


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.optimizer = None
        self.scheduler = None
        self.eval_fn = None
        self.measure_train = {'measure':'', 'values':[]}
        self.measure_val = {'measure':'', 'values':[]}
        self.measure_test = {'measure':'', 'values':[]}

    def _init_weight(self):
        pass

    def _set_optimizer(self, config):
        optimizer = getattr(optim, config.method)
        self.optimizer = optimizer(self.parameters(),
                                   lr=config.learning_rate,
                                   weight_decay=config.regulation)

        print(f'[BaseModel] method = {config.method}')
        print(f'[BaseModel] regulation = {config.regulation}')
        print(f'[BaseModel] learning_rate = {config.learning_rate}')

    def _set_scheduler(self, config):
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                     lr_lambda=lambda epoch: config.learning_decay ** epoch,
                                                     last_epoch=-1)

        print(f'[BaseModel] learning_rate_decay = {config.learning_decay}')

    def _set_loss_fn(self):
        pass

    def _get_eval_fn(self, method):
        self.measure_val['measure'] = method
        self.measure_test['measure'] = method

        measure = getattr(utils_evaluation, method.lower())

        return measure

    def fit(self, *input):
        pass

    def forward(self, *input):
        pass

    def backward(self, *input):
        pass

    def predict(self, *input):
        pass

    def evaluate(self, *input):
        pass

    def test(self, *input):
        pass

    def load_weight(self, model):
        self.load_state_dict(model)

    def get_weight(self):
        return self.state_dict()

    def draw_plot(self):
        measure_train = self.measure_train['values']
        measure_val = self.measure_val['values']
        measure_test = self.measure_test['values']

        epoch = len(measure_train)
        x = range(1, epoch+1)

        y_min = min(measure_val + measure_test)
        y_max = max(measure_val + measure_test)

        fig, axs = plt.subplots(1, 2, figsize=(12,4))
        fig.subplots_adjust(wspace=0.3)

        loss_ax = axs[0]
        loss_ax.plot(x, self.measure_train['values'], 'royalblue', marker='.',  label=f'train {self.measure_train["measure"]}')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel(f'{self.measure_train["measure"]}')
        loss_ax.legend(loc='upper right')
        loss_ax.set_title(f'{self.measure_train["measure"]} for trainset')
        loss_ax.grid(axis='y', linestyle=':')

        eval_ax = axs[1]
        eval_ax.plot(x, self.measure_val['values'], 'mediumslateblue', marker='.',  label=f'validation {self.measure_val["measure"]}')
        eval_ax.hlines(self.measure_test['values'], 1, epoch, 'indigo', linestyles='dashed',  label='test performance')
        eval_ax.set_ylim(y_min-np.std(measure_val), y_max+np.std(measure_val))
        eval_ax.set_xlabel('epoch')
        eval_ax.set_ylabel(f'{self.measure_val["measure"]}')
        eval_ax.legend(loc='lower right')
        eval_ax.set_title(f'{self.measure_val["measure"]} for validation/testset')
        eval_ax.grid(axis='y', linestyle=':')

        return fig
