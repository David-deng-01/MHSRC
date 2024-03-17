from .trainer import Trainer


def get_misa_scheduler(cls: Trainer):
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(cls.optimizer, gamma=0.5)
    # return lr_scheduler
    return None

def get_mcwf_scheduler(cls: Trainer):
    return None


def get_cubemlp_scheduler(cls: Trainer):
    return None

def get_unimse_scheduler(cls: Trainer):
    return None

def get_cmgcn_scheduler(cls: Trainer):
    return None

def get_dip_scheduler(cls: Trainer):
    return None

def get_sks_scheduler(cls: Trainer):
    return None

def get_simmmdg_scheduler(cls: Trainer):
    return None