class OptimizerConfig:
    method: str             = 'Adam'
    regulation: float       = 1e-5
    learning_rate: float    = 0.01
    learning_decay: float   = 0.95


class SVDConfig:
    arguments: list = ['n_user', 'n_item', 'mu']
    evaluation: str = 'MSE'
    dim_latent: int = 16


class SVDppConfig:
    arguments: list = ['n_user', 'n_item', 'mu']
    evaluation: str = 'MSE'
    dim_latent: int = 16


class GMFConfig:
    arguments: list = ['n_user', 'n_item']
    evaluation: str = 'HR'
    dim_latent: int = 8
    k: int          = 10
    sample: int     = 100


class MLPConfig:
    arguments: list = ['n_user', 'n_item']
    evaluation: str = 'HR'
    dim_latent: int = 16
    k: int          = 10
    sample: int     = 100


class NeuMFConfig:
    arguments: list     = ['n_user', 'n_item']
    evaluation: str     = 'HR'
    dim_latent_gmf: int = 8
    dim_latent_mlp: int = 16
    alpha: float        = 0.5
    k: int              = 10
    sample: int         = 100


class AutoRecConfig:
    arguments: list     = ['n_user', 'n_item']
    evaluation: str     = 'MSE'
    dim_latent: int     = 300
    fill_na_as: float   = 3.0


class CDAEConfig:
    arguments: list      = ['n_user', 'n_item']
    evaluation: str      = 'MAP'
    k: int               = 10
    dim_latent: int      = 100
    corrupt_ratio: float = 0.6


class BPRConfig:
    arguments: list     = ['n_user', 'n_item']
    evaluation: str     = 'AUC'
    dim_latent: int     = 16


class NGCFConfig:
    arguments: list     = ['n_user', 'n_item']
    evaluation: str     = 'NDCG'
    k: int              = 20
    dim_embedding: int  = 16
    num_layer: int      = 3
    dropout_msg: float  = 0.1
    dropout_node: float = 0.0
