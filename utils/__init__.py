from .eval import *
from .optimizers import *
from .schedulers import *

EVAL_FN = {
    'mcwf': mcwf_eval_fn,
    'misa': misa_eval_fn,
    'cubemlp': cubemlp_eval_fn,
    'unimse': unimse_eval_fn,
    'dip': dip_eval_fn,
}

TEST_FN = {
    'mcwf': mcwf_eval_fn,
    'misa': misa_eval_fn,
    'cubemlp': cubemlp_eval_fn,
    'unimse': unimse_eval_fn,
    'dip': dip_eval_fn,
}

ML_EVAL_FN = {
    'mcwf': mcwfml_eval_fn,
    'cubemlp': cubemlpml_eval_fn,
    'dip': dipmlpml_eval_fn,
    'misa': misaml_eval_fn,
    'sks': sks_eval_fn
}

ML_TEST_FN = {
    'mcwf': mcwfml_eval_fn,
    'cubemlp': cubemlpml_eval_fn,
    'dip': dipmlpml_eval_fn,
    'misa': misaml_eval_fn,
    'sks': sks_eval_fn
}

CLASSIFIER_EVAL_FN = {
    'mcwf': mcwfforcl_eval_fn,
    'cubemlp': cubemlpforcl_eval_fn,
    # 'dip': dipmlpforcl_eval_fn,
    'misa': misaforcl_eval_fn,
}

CLASSIFIER_TEST_FN = {
    'mcwf': mcwfforcl_eval_fn,
    'cubemlp': cubemlpforcl_eval_fn,
    # 'dip': dipmlpforcl_eval_fn,
    'misa': misaforcl_eval_fn,
}

OPTIMIZER_FN = {
    'mcwf': None,
    'misa': None,
    'cubemlp': None,
    'unimse': None,
    'dip': None,
    'sks': None,
}

SCHEDULER_FN = {
    'mcwf': None,
    'misa': None,
    'cubemlp': None,
    'unimse': None,
    'dip': None,
    'sks': None,
}


ML_OPTIMIZER_FN = {
    'mcwf': None,
    'misa': None,
    'cubemlp': None,
    'unimse': None,
    'dip': None,
    'sks': None,
}

ML_SCHEDULER_FN = {
    'mcwf': None,
    'misa': None,
    'cubemlp': None,
    'unimse': None,
    'dip': None,
    'sks': None,
}
