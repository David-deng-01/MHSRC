from .trainer import Trainer


def get_misa_optimizer(cls: Trainer):
    # optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.config['lr'])
    # return optimizer
    return None


def get_mcwf_optimizer(cls: Trainer):
    return None


def get_cubemlp_optimizer(cls: Trainer):
    return None


def get_unimse_optimizer(cls: Trainer):
    return None


def get_cmgcn_optimizer(cls: Trainer):
    return None


def get_dip_optimizer(cls: Trainer):
    return None


def get_sks_optimizer(cls: Trainer):
    return None


def get_simmmdg_optimizer(cls: Trainer):
    return None
