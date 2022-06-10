class PreprocessConfig:
    min_item_per_user: int      = 10 # 10
    min_user_per_item: int      = 1
    implicit: bool              = True
    implicit_threshold: float   = 0


class DataSplitConfig:
    method: str             = 'holdout'  # holdout, leave_k_out
    validation_ratio: float = 0.1
    test_ratio: float       = 0.2
    leave_k: int            = 1
    shuffle: bool           = True
    seed: int               = 2013


class DatasetConfig:
    method: str             = 'pairwise' # pointwise / pairwise / matrix
    num_negative: int       = 2
    between_observed: bool  = False
    batch: int              = 512


class ModelConfig:
    algorithm: str          = 'NGCF'
    epochs: int             = 30
    load_dir: list          = []
    print_step: int         = 1


class SaveConfig:
    save_dir: str       = 'output'
    save_meta: bool     = False
    save_model: bool    = False
    save_plot: bool     = False
    save_log: bool      = False


class TrainerConfig:
    preprocess_config = PreprocessConfig
    data_split_config = DataSplitConfig
    dataset_config = DatasetConfig
    model_config = ModelConfig
    save_config = SaveConfig
