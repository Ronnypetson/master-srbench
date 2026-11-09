import os

import torch
from nicoboco.canon import db_recompute_comb_size_enumeration
from nicoboco.constraints import StructuralConstraints
from nicoboco.nnet import ConditionalGenericCausalTransformer
from nicoboco.primitives import Primitives
from nicoboco.regressor import CanonExpressionDataset, SymbolicRegressor

from .customop import Add, Cos, Div, Log, Mul, Pow, Sin, Sub


def model(est, X=None):
    return est.expression


def param_sampler(lo=-8.0, hi=8.0):
    assert hi > lo
    return lo + (hi - lo) * torch.rand(1, dtype=torch.float64)


max_expr_len = 11
effective_max_expr_len = 2 * max_expr_len - 1
maxnvar = 10

op_list = [
    Add(),
    Sub(),
    Mul(),
    Div(),
    Pow(),
    Log(),
    Sin(),
    Cos(),
]
varlist = [f"x{idx}" for idx in range(maxnvar)]
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
    max_op_counts={},
)

arr_domain = torch.rand((6, 10), dtype=torch.float64) * 20 - 10
scratch_root = "tmp"
enum_work_dir = os.path.join(scratch_root, "nicoboco_enum_tmp")
canon_expr, _, canon_ptrs = db_recompute_comb_size_enumeration(
    primitives,
    constraints,
    arr_domain,
    commit_every=1000000,
    work_dir=enum_work_dir,
)
print(canon_ptrs)

canon_max = max(canon_ptrs.keys())
assert canon_max == max_expr_len
num_points = 128
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
    max_seq_len=(effective_max_expr_len + ctr_nvar),
    num_in_size=num_points,
).to(device)

canon_dataset = CanonExpressionDataset(canon_expr, canon_ptrs)

est = SymbolicRegressor(
    primitives,
    policy_model,
    canon_dataset,
    num_points=num_points,
    numeric_padding=0.0,
    const_prob=(1 / 7),
    random_sample=True,
    tgt_noise_level=0.0,
    batch_size=96,
    num_workers=0,
    peak_lr=1e-5,
    warmup_steps=2000,
    total_iter=8000,
    weight_decay=0.01,
    hyp_per_turn=100,
    max_turns=10000,
    n_pts_fit=256,
    eps=1e-7,
    test_eps=1.0, # 1e-2 for ground-truth problems
    n_inits=400,
    fit_params=True,
    full_it=10,
    secret_ids=None,
    verbose=True,
    temp_sched=None,
    subsample_data=True,
    device=device,
)
