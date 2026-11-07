import os
import torch # type: ignore

from nicoboco.regressor import SymbolicRegressor # type: ignore
from nicoboco.primitives import Primitives # type: ignore
from nicoboco.constraints import StructuralConstraints # type: ignore
from nicoboco.canon import db_recompute_comb_size_enumeration # type: ignore
from nicoboco.nnet import ConditionalGenericCausalTransformer # type: ignore

from .customop import Add, Sub, Mul, Div, Pow, Log, Sin, Cos # type: ignore


def model(est, X=None):
    return est.expression

def param_sampler(lo=-8.0, hi=8.0):
    assert hi > lo
    return lo + (hi - lo) * torch.rand(1, dtype=torch.float64)

max_expr_len = 7

op_list = [
    Add(), Sub(), Mul(), Div(), Pow(), Log(), # Binary
    Sin(), Cos(), # Unary
]
varlist = [f"x{idx}" for idx in range(20)]
primitives = Primitives(
    operators=op_list,
    varnames=varlist,
    param_sampler=param_sampler,
)

constraints = StructuralConstraints(
    max_complexity=max_expr_len,
    nested_rules={
        "Sin": {"Sin": 0, "Cos": 0, "Pow": 1, "Log": 0},
        "Cos": {"Sin": 0, "Cos": 0, "Pow": 1, "Log": 0},
        "Pow": {"Pow": 0, "Log": 1, "Sin": 1, "Cos": 1},
        "Log": {"Pow": 1, "Log": 0, "Sin": 0, "Cos": 0},
    },
    max_op_counts={}
)

arr_domain = torch.rand((6, 10), dtype=torch.float64) * 20 - 10
scratch_root = "tmp"
enum_work_dir = os.path.join(scratch_root, f"nicoboco_enum_tmp")
canon_expr, _, canon_ptrs = db_recompute_comb_size_enumeration(
    primitives, constraints, arr_domain, work_dir=enum_work_dir,
)
print(canon_ptrs)

canon_max = max(canon_ptrs.keys())
num_points = 64
d_model = 1024
nhead = 8
num_layers = 16
ctr_nvar = len(primitives.varnames)
vocab_size = len(primitives.vocab)

device = "cuda:0"

policy_model = ConditionalGenericCausalTransformer(
    vocab_size,
    d_model,
    nhead,
    num_layers,
    max_seq_len=(canon_max + ctr_nvar),
    num_in_size=num_points,
).to(device)

est = SymbolicRegressor(
    primitives,
    policy_model,
    canon_expr,
    canon_ptrs,
    num_points = 64,
    numeric_padding = 0.0,
    const_prob = 1/7,
    random_sample = True,
    tgt_noise_level = 0.0,
    batch_size = 96,
    num_workers = 8,
    peak_lr = 1e-5,
    warmup_steps = 2000,
    total_iter = 8000,
    weight_decay = 0.01,
    hyp_per_turn = 1000,
    max_turns = 1000,
    n_pts_fit = 128,
    eps = 1e-7, # 1e-8,
    test_eps = 1e-2, # 1e-2,
    n_inits = 400, # 400,
    fit_params = True,
    full_it = 10,
    secret_ids = None,
    verbose = True,
    temp_sched = None,
    subsample_data = True,
    device = device,
)
