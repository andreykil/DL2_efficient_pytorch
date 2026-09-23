import math
import os
import time
import urllib.request

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

TRAIN_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

BATCH_SIZE = 8
VAL_BATCH_SIZE = 1024
N_PROMPTS = 128
D_MODEL = 128
N_CYCLES = 6
EPOCHS = 5
SEED = 0


def stacked_layer_norm(x, weight, bias):
    x = F.layer_norm(x, (x.shape[-1],))
    shape = (weight.shape[0],) + (1,) * (x.ndim - 2) + (weight.shape[1],)
    return (x * weight.view(shape) + bias.view(shape)).to(torch.float16)


class TromptCells(nn.Module):
    def __init__(self, n_cycles, n_columns):
        super().__init__()
        # Объединяем несколько циклов в batched-тензоры
        k, c, d, p = n_cycles, n_columns, D_MODEL, N_PROMPTS
        self.feature_weight = nn.Parameter(torch.empty(k, c, d))
        self.feature_bias = nn.Parameter(torch.empty(k, c, d))
        self.ln_emb_weight = nn.Parameter(torch.ones(k, d))
        self.ln_emb_bias = nn.Parameter(torch.zeros(k, d))
        self.ln_col_weight = nn.Parameter(torch.ones(k, d))
        self.ln_col_bias = nn.Parameter(torch.zeros(k, d))
        self.ln_prompt_weight = nn.Parameter(torch.ones(k, d))
        self.ln_prompt_bias = nn.Parameter(torch.zeros(k, d))
        self.imp_weight = nn.Parameter(torch.empty(k, d, 2 * d))
        self.imp_bias = nn.Parameter(torch.empty(k, d))
        self.emb_column = nn.Parameter(torch.empty(k, c, d))
        self.emb_prompt = nn.Parameter(torch.empty(k, p, d))
        self.expand_weight = nn.Parameter(torch.empty(k, p))
        self.expand_bias = nn.Parameter(torch.empty(k, p))
        self.reset_parameters()

    def reset_parameters(self):
        a = D_MODEL**-0.5
        b = (2 * D_MODEL) ** -0.5
        nn.init.uniform_(self.feature_weight, -a, a)
        nn.init.uniform_(self.feature_bias, -a, a)
        nn.init.uniform_(self.imp_weight, -b, b)
        nn.init.uniform_(self.imp_bias, -b, b)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)
        nn.init.uniform_(self.expand_weight, -1.0, 1.0)
        nn.init.uniform_(self.expand_bias, -1.0, 1.0)

    def forward(self, x, prompt):
        # Строим эмбеддинги признаков
        fw = self.feature_weight.to(torch.float16)
        fb = self.feature_bias.to(torch.float16)
        x_emb = F.relu(x[None, :, :, None] * fw[:, None] + fb[:, None])
        x_emb = stacked_layer_norm(x_emb, self.ln_emb_weight, self.ln_emb_bias)

        # Разделяем Linear на два умножения, чтобы не делать concat
        norm_prompt = stacked_layer_norm(
            self.emb_prompt, self.ln_prompt_weight, self.ln_prompt_bias
        )
        w_prompt, w_prev = self.imp_weight.split(D_MODEL, dim=-1)
        x_prompt = (
            norm_prompt @ w_prompt.transpose(-1, -2)
            + prompt[None] @ w_prev.transpose(-1, -2)
            + self.imp_bias[:, None].to(torch.float16)
            + self.emb_prompt.to(torch.float16)
        )
        x_column = stacked_layer_norm(
            self.emb_column, self.ln_col_weight, self.ln_col_bias
        )
        mask = torch.softmax(torch.bmm(x_prompt, x_column.transpose(1, 2)), dim=-1)

        # Expansion-блок линейный, поэтому большой промежуточный тензор не нужен
        weighted = torch.matmul(mask[:, None], x_emb)
        return (
            weighted,
            (1 + self.expand_weight).to(weighted.dtype),
            self.expand_bias.to(weighted.dtype),
        )


class TromptShard(nn.Module):
    def __init__(self, n_cycles, n_columns, rank):
        super().__init__()
        torch.manual_seed(SEED + rank + 1)
        self.cells = TromptCells(n_cycles, n_columns)

        torch.manual_seed(SEED)
        self.prompt = nn.Parameter(torch.empty(N_PROMPTS, D_MODEL))
        self.dense0 = nn.Linear(D_MODEL, 1)
        self.dense1 = nn.Linear(D_MODEL, D_MODEL)
        self.ln = nn.LayerNorm(D_MODEL)
        self.dense_out = nn.Linear(D_MODEL, 1)
        nn.init.normal_(self.prompt, std=0.01)

    def shared_parameters(self):
        return [
            self.prompt,
            *self.dense0.parameters(),
            *self.dense1.parameters(),
            *self.ln.parameters(),
            *self.dense_out.parameters(),
        ]

    def forward(self, x):
        # Prompt одинаков для всего батча, поэтому не копируем его по оси batch
        weighted, scale, expand_bias = self.cells(x, self.prompt)

        # Переносим линейный expansion через первый downstream-слой
        logits = F.linear(weighted, self.dense0.weight, None).squeeze(-1)
        logits = (
            logits * scale[:, None]
            + expand_bias[:, None] * self.dense0.weight.sum()
            + self.dense0.bias
        )
        pw = torch.softmax(logits, dim=-1)

        x = torch.matmul((pw * scale[:, None]).unsqueeze(-2), weighted).squeeze(-2)
        x = x + (pw * expand_bias[:, None]).sum(dim=-1, keepdim=True)
        x = self.ln(F.relu(self.dense1(x)))
        return self.dense_out(x).squeeze(-1).transpose(0, 1)


def download(url):
    path = url.rsplit("/", 1)[-1]
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load_data(device, rank):
    if rank == 0:
        download(TRAIN_DATA)
        download(VAL_DATA)
    dist.barrier()

    train_x, train_y = torch.load(download(TRAIN_DATA), weights_only=True)
    val_x, val_y = torch.load(download(VAL_DATA), weights_only=True)
    for t in (train_x, train_y, val_x, val_y):
        t.nan_to_num_()

    y_mean, y_std = train_y.mean(), train_y.std()
    train_y.sub_(y_mean).div_(y_std)
    # Заранее переносим весь датасет на GPU
    return (
        train_x.to(device, dtype=torch.float16),
        train_y.to(device),
        val_x.to(device, dtype=torch.float16),
        val_y.to(device),
        y_mean.to(device),
        y_std.to(device),
    )


def shuffled_indices(size, epoch, device):
    g = torch.Generator().manual_seed(SEED + epoch)
    return torch.randperm(size, generator=g).to(device)


def make_grad_bucket(params):
    bucket = torch.zeros(
        sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype
    )
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = bucket[offset : offset + n].view_as(p)
        offset += n
    return bucket


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size != 2 or not torch.cuda.is_available():
        raise RuntimeError(
            "Run on Kaggle T4 x2 with: torchrun --standalone --nproc_per_node=2 train.py"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    train_x, train_y, val_x, val_y, y_mean, y_std = load_data(device, rank)
    # Каждая GPU считает половину независимых циклов модели
    model = TromptShard(N_CYCLES // world_size, train_x.shape[1], rank).to(device)

    shared = model.shared_parameters()
    for p in shared:
        dist.broadcast(p.data, src=0)
    local = list(model.cells.parameters())
    shared_grad = make_grad_bucket(shared)

    # Компилируем модель через torch.compile
    model.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
    # Используем fused-версию AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5, fused=True
    )
    # AMP выполняет основные вычисления в float16
    scaler = torch.amp.GradScaler("cuda")
    global_batch = BATCH_SIZE * world_size

    def zero_grad():
        # Храним общие градиенты как части одного буфера
        shared_grad.zero_()
        for p in local:
            p.grad = None

    def step(ids, update=True):
        zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(train_x[ids])
            target = train_y[ids, None].expand_as(pred)
            loss = F.mse_loss(pred, target) / world_size
        scaler.scale(loss).backward()

        # Синхронизируем общий буфер одним вызовом NCCL
        dist.all_reduce(shared_grad, op=dist.ReduceOp.SUM)
        if update:
            scaler.step(optimizer)
            scaler.update()

    # warmup до начала замера
    warmup_ids = shuffled_indices(len(train_x), 0, device)[:global_batch]
    for _ in range(3):
        step(warmup_ids, update=False)
    zero_grad()
    torch.cuda.synchronize(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        indices = shuffled_indices(len(train_x), epoch, device)
        starts = range(0, len(indices), global_batch)
        if rank == 0:
            starts = tqdm(
                starts,
                total=math.ceil(len(indices) / global_batch),
                desc=f"Epoch {epoch:02}",
                mininterval=1.0,
                dynamic_ncols=True,
            )

        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for start in starts:
            step(indices[start : start + global_batch])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        if rank == 0:
            print(f">>> Epoch {epoch:02}")
            print(f"Training throughput = {len(indices) / elapsed:,.0f} objects/s", flush=True)

        dist.barrier()
        model.eval()
        error = torch.zeros((), device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(val_x), VAL_BATCH_SIZE):
                pred = model(val_x[start : start + VAL_BATCH_SIZE]).float().sum(-1)
                dist.all_reduce(pred, op=dist.ReduceOp.SUM)
                pred = pred / N_CYCLES * y_std + y_mean
                if rank == 0:
                    error += (pred - val_y[start : start + VAL_BATCH_SIZE]).abs().sum()

        if rank == 0:
            print(f"Validation MAE = {(error / len(val_x)).item():.5f}")
            print(">>>\n", flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
