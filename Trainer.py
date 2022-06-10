import argparse
import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import datetime

from config import TrainerConfig
import Generator
import Models


class Trainer:
    def __init__(self, data_path, user_colname, item_colname, rating_colname, config):

        self.index_to_id_user = None
        self.index_to_id_item = None
        self.id_to_index_user = None
        self.id_to_index_item = None
        self.num_user = None
        self.num_item = None
        self.mu = None
        self.model = None
        self.model_arguments = None

        self.support_ext = ['csv']
        self.argument_dict = dict.fromkeys(['n_user', 'n_item', 'mu'])

        self.preprocess_config = config.preprocess_config
        self.data_split_config = config.data_split_config
        self.dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.save_config = config.save_config

        self.data_path = data_path
        self.user_colname = user_colname
        self.item_colname = item_colname
        self.rating_colname = rating_colname

        self._init_model()

    def fit(self):
        raw_data = self._load_data()
        data = self._preprocess(raw_data)
        train_data, val_data, test_data = self._split(data)
        train_loader = self._generate_loader(train_data)
        self._train(train_data, val_data, test_data, train_loader)

        self._save_meta()

    def _load_data(self):
        print(f'\n[Trainer]'+' Loading Data '.center(100, '='))
        data_type = os.path.splitext(self.data_path)[-1][1:]
        if data_type not in self.support_ext:
            print(f'[Trainer] Unsupported data type. Try other extension({", ".join(self.support_ext)}).')
            data = None

        if data_type == "csv":
            raw = pd.read_csv(self.data_path)
            data = pd.DataFrame({'user_id':raw[self.user_colname],
                                 'item_id':raw[self.item_colname],
                                 'rating':raw[self.rating_colname]})
        print(f'[Trainer] data path: {self.data_path}')

        return data

    def _preprocess(self, data):
        print(f'\n[Trainer]'+' Preprocess Data '.center(100, '='))
        pre_length = len(data)

        # if implicit is True, transform explicit feedback to implicit feedback with threshold
        if self.preprocess_config.implicit:
            is_upper_thr = data['rating'] >= self.preprocess_config.implicit_threshold
            remain_idx = is_upper_thr[is_upper_thr==True].index
            data = data.loc[data.index.isin(remain_idx)==True]
            data['rating'].values.fill(1)

        # remove user/item which count lower than minimum threshold
        while True:
            if (min(data.groupby('item_id').count()['user_id']) >= self.preprocess_config.min_item_per_user) and\
               (min(data.groupby('user_id').count()['item_id']) >= self.preprocess_config.min_user_per_item):
                break

            lower_item_per_user = data.groupby('user_id').count()['item_id']<self.preprocess_config.min_user_per_item
            remove_user_ids = lower_item_per_user[lower_item_per_user==True].index
            data = data[data.user_id.isin(remove_user_ids)==False]

            lower_user_per_item = data.groupby('item_id').count()['user_id']<self.preprocess_config.min_item_per_user
            remove_item_ids = lower_user_per_item[lower_user_per_item==True].index
            data = data[data.item_id.isin(remove_item_ids)==False]

        # convert id to index for user/item
        user_ids = data['user_id'].unique().tolist()
        item_ids = data['item_id'].unique().tolist()

        self.num_user = len(user_ids)
        self.num_item = len(item_ids)

        self.index_to_id_user = {i :user_ids[i] for i in range(self.num_user)}
        self.index_to_id_item = {i :item_ids[i] for i in range(self.num_item)}
        self.id_to_index_user = {user_ids[i] :i for i in range(self.num_user)}
        self.id_to_index_item = {item_ids[i] :i for i in range(self.num_item)}

        data['user_id'] = data['user_id'].apply(lambda x: self.id_to_index_user[x])
        data['item_id'] = data['item_id'].apply(lambda x: self.id_to_index_item[x])

        # @TODO center, std normalizatio 추가하기 근데 trainset 정보만 가져야되니까 다른 곳에서 해야될듯

        now_length = len(data)
        print(f'[Trainer] # of observation: {pre_length} -> {now_length}')
        print(f'[Trainer] min_item_per_user = {self.preprocess_config.min_item_per_user}')
        print(f'[Trainer] min_user_per_item = {self.preprocess_config.min_user_per_item}')
        print(f'[Trainer] implicit = {self.preprocess_config.implicit}')
        print(f'[Trainer] implicit_threshold = {self.preprocess_config.implicit_threshold}')

        return data

    def _split(self, data):
        print(f'\n[Trainer]'+' Split Data '.center(100, '='))

        train, validation, test = None, None, None

        if self.data_split_config.method == 'holdout':
            shuffle = self.data_split_config.shuffle

            val_ratio = self.data_split_config.validation_ratio
            test_ratio = self.data_split_config.test_ratio
            train, others = train_test_split(data,
                                             test_size=val_ratio+test_ratio,
                                             shuffle=shuffle,
                                             random_state=self.data_split_config.seed)

            validation, test = train_test_split(others,
                                                test_size=test_ratio/(val_ratio+test_ratio),
                                                shuffle=shuffle,
                                                random_state=self.data_split_config.seed)

        elif self.data_split_config.method == 'leave_k_out':
            leave_k = self.data_split_config.leave_k
            seed = self.data_split_config.seed

            test = data.groupby('user_id').sample(leave_k, random_state=seed)
            train_ = data.loc[data.index.isin(test.index)==False]
            validation = train_.groupby('user_id').sample(leave_k, random_state=seed)
            train = train_.loc[train_.index.isin(validation.index) == False]

        else:
            print('unsupported split method')
            exit(0)

        print(f'[Trainer] # of observation(train/val/test): {len(train)}, {len(validation)}, {len(test)}')
        print(f'[Trainer] method = {self.data_split_config.method}')
        if self.data_split_config.method == 'holdout':
            print(f'[Trainer] validation_ratio = {self.data_split_config.validation_ratio}')
            print(f'[Trainer] test_ratio = {self.data_split_config.test_ratio}')
        elif self.data_split_config.method == 'leave_k':
            print(f'[Trainer] leave_k_out = {self.data_split_config.leave_k}')

        return train, validation, test

    def _generate_loader(self, train_data):
        print(f'\n[Trainer]'+' Generate DataLoader '.center(100, '='))

        generator = getattr(Generator, self.dataset_config.method.capitalize()+'Generator')
        dataset = generator(train_data=train_data,
                            n_user=self.num_user,
                            n_item=self.num_item,
                            config=self.dataset_config)
        train_loader = DataLoader(dataset, batch_size=self.dataset_config.batch)

        print(f'[Trainer] method = {self.dataset_config.method}')
        print(f'[Trainer] num_negative = {self.dataset_config.num_negative}')
        if self.dataset_config.num_negative >0:
            print(f'[Trainer] between_observed = {self.dataset_config.between_observed}')
        print(f'[Trainer] batch = {self.dataset_config.batch}')

        return train_loader

    def _train(self, train_data, val_data, test_data, train_loader):
        print(f'\n[Trainer]'+' Set model '.center(100, '='))
        print(f'[Trainer] algorithm = {self.model_config.algorithm}')

        self._update_args_dict(mu=train_data['rating'].mean())
        model_input = self._get_input()
        self.model = self.model(model_input, config_train=self.model_config)
        for dir in self.model_config.load_dir:
            pre_trained = torch.load(dir)
            now_weight = self.model.get_weight()
            now_weight.update(pre_trained)
            self.model.load_weight(now_weight)
            print(f'[Trainer] load_dir = {", ".join(self.model_config.load_dir)}')
        self.model.fit(train_loader=train_loader, val_data=val_data)
        self.model.test(train_loader=train_loader, test_data=test_data)
        # test_result = self.model.evaluate(dataloader=test_loader)

    def _init_model(self):
        self.model = getattr(Models, self.model_config.algorithm)
        config = getattr(Models.config, self.model_config.algorithm+'Config')
        self.model_arguments = config.arguments

    def _get_input(self):
        return {k: self.argument_dict[k] for k in self.model_arguments}

    def _update_args_dict(self, mu):
        self.argument_dict = {'n_user': self.num_user,
                              'n_item': self.num_item,
                              'mu': mu}

    def _log(self, train_loss, val_loss, test_loss):
        msg = "[loss]\n"
        msg += f"train = {train_loss:.4f}, validation = {val_loss:.4f}, test = {test_loss:.4f}\n"

        return msg

    def _save_meta(self):
        print(f'\n[Trainer]'+' Save Output '.center(100, '='))

        if self.save_config.save_dir != '':
            # @TODO: tree구조 mkdir
            if not os.path.exists(self.save_config.save_dir ):
                os.mkdir(self.save_config.save_dir )

        now = datetime.datetime.now()
        now = now.strftime('%m%d%y_%H%M')
        save_name = self.model_config.algorithm + '_' + now
        save_name = os.path.join(self.save_config.save_dir, save_name)

        if self.save_config.save_meta:
            # @TODO
            pass

        if self.save_config.save_model:
            weight = self.model.get_weight()
            torch.save(weight, save_name + '.pth')
            print(f'[Trainer] saved model at {save_name + ".pth"}')

        if self.save_config.save_plot:
            figure = self.model.draw_plot()
            figure.savefig(save_name + '.png')
            # figure.close()
            print(f'[Trainer] saved plot at {save_name + ".png"}')

        if self.save_config.save_log:
            log = self._log(train_loss=self.model.measure_train['values'][-1].tolist(),
                            val_loss=self.model.measure_train['values'][-1].tolist(),
                            test_loss=self.model.measure_train['values'][-1].tolist())
            with open(save_name + '.txt', "w") as f:
                f.write(log)


def main(args):

    trainer = Trainer(data_path=args.data_path,
                      user_colname=args.user_colname,
                      item_colname=args.item_colname,
                      rating_colname=args.rating_colname,
                      config=TrainerConfig)
    trainer.fit()
    sys.exit()


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Argument for training.')

    parser.add_argument('--data_path', type=str, help='path including extension')
    parser.add_argument('--user_colname', type=str, help='column name for user')
    parser.add_argument('--item_colname', type=str, help='column name for item')
    parser.add_argument('--rating_colname', type=str, help='column name for rating')

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True

DATA_PATH = 'data/movie_lens/ratings.csv'
USER_COLNAME = "userId"
ITEM_COLNAME = "movieId"
RATING_COLNAME = "rating"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--data_path", DATA_PATH])
            sys.argv.extend(["--user_colname", USER_COLNAME])
            sys.argv.extend(["--item_colname", ITEM_COLNAME])
            sys.argv.extend(["--rating_colname", RATING_COLNAME])

        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))