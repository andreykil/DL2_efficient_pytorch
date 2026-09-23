import time
import urllib.request
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
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


class TromptCell(nn.Module):
    def __init__(self, n_columns: int, n_prompts: int, d_model: int):
        super().__init__()

        limit = d_model**-0.5

        self.feature_emb_weight = mx.random.uniform(
            low=-limit, high=limit, shape=(n_columns, d_model)
        )
        self.feature_emb_bias = mx.random.uniform(
            low=-limit, high=limit, shape=(n_columns, d_model)
        )
        self.ln_emb = nn.LayerNorm(d_model)

        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = mx.random.normal(shape=(n_columns, d_model)) * 0.01
        self.emb_prompt = mx.random.normal(shape=(n_prompts, d_model)) * 0.01

        self.dense_expand = nn.Linear(1, n_prompts)

    def __call__(self, x: mx.array, previous: mx.array):
        # Строим эмбеддинги признаков
        x_emb = x[..., None] * self.feature_emb_weight + self.feature_emb_bias
        x_emb = self.ln_emb(nn.relu(x_emb))

        # Разделяем Linear на два умножения, чтобы не делать concat
        d_model = self.emb_prompt.shape[-1]
        prompt_weight = self.dense_imp.weight[:, :d_model]
        previous_weight = self.dense_imp.weight[:, d_model:]

        prompt_part = self.ln_prompt(self.emb_prompt) @ prompt_weight.T
        prompt_part = prompt_part + self.dense_imp.bias

        x_prompt = (
            previous @ previous_weight.T
            + prompt_part[None, ...]
            + self.emb_prompt[None, ...]
        )

        x_column = self.ln_col(self.emb_column)
        mask = mx.softmax(x_prompt @ x_column.T, axis=-1)

        # Expansion-блок линейный, поэтому большой промежуточный тензор не нужен
        weighted = mask @ x_emb
        scale = mx.squeeze(self.dense_expand.weight, axis=-1) + 1
        bias = self.dense_expand.bias

        return weighted, scale, bias


class TromptDownstream(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def __call__(
        self,
        weighted: mx.array,
        scale: mx.array,
        bias: mx.array,
    ) -> mx.array:
        # Переносим линейный expansion через первый downstream-слой
        logits = mx.squeeze(weighted @ self.dense0.weight.T, axis=-1) * scale
        logits += bias * mx.sum(self.dense0.weight) + self.dense0.bias

        prompt_weight = mx.softmax(logits, axis=-1)

        x = mx.sum(
            prompt_weight[..., None] * weighted * scale[..., None],
            axis=1,
        )
        x += mx.sum(prompt_weight * bias, axis=1, keepdims=True)

        x = self.dense1(x)
        x = self.ln(nn.relu(x))
        return self.dense_out(x)


class Trompt(nn.Module):
    def __init__(
        self,
        n_columns: int,
        n_prompts: int,
        d_model: int,
        n_cycles: int,
    ):
        super().__init__()

        self.tcells = [
            TromptCell(n_columns, n_prompts, d_model)
            for _ in range(n_cycles)
        ]
        self.tdown = TromptDownstream(d_model)
        self.prompt = mx.random.normal(shape=(n_prompts, d_model)) * 0.01

    def __call__(self, x: mx.array) -> mx.array:
        # Prompt одинаков для всего батча, поэтому не копируем его по оси batch
        outputs = [
            self.tdown(*cell(x, self.prompt))
            for cell in self.tcells
        ]
        return mx.concatenate(outputs, axis=1)


def download(url: str) -> Path:
    path = Path(__file__).parent / url.rsplit("/", 1)[-1]

    if not path.exists():
        print(f"Downloading {path.name} ...")
        urllib.request.urlretrieve(url, path)

    return path


def load_data():
    # Torch нужен только для чтения файлов с тензорами
    train_x, train_y = torch.load(
        download(TRAIN_DATA),
        map_location="cpu",
        weights_only=True,
    )
    val_x, val_y = torch.load(
        download(VAL_DATA),
        map_location="cpu",
        weights_only=True,
    )

    train_x, train_y, val_x, val_y = (
        torch.nan_to_num(t)
        for t in (train_x, train_y, val_x, val_y)
    )

    y_mean = train_y.mean()
    y_std = train_y.std()
    train_y = (train_y - y_mean) / y_std

    tensors = train_x, train_y, val_x, val_y, y_mean, y_std
    return tuple(mx.array(t.numpy()) for t in tensors)


def main():
    mx.random.seed(SEED)

    train_x, train_y, val_x, val_y, y_mean, y_std = load_data()

    model = Trompt(
        n_columns=train_x.shape[1],
        n_prompts=N_PROMPTS,
        d_model=D_MODEL,
        n_cycles=N_CYCLES,
    )

    optimizer = optim.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-5,
        bias_correction=True,
    )
    optimizer.init(model.trainable_parameters())

    def loss_fn(model, x, y):
        prediction = model(x)
        target = y[:, None]
        return mx.mean(mx.square(prediction - target))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # MLX вычисляет лениво, поэтому состояния явно передаем в скомпилированный шаг
    state = [model.state, optimizer.state]

    # Компилируем forward, backward и обновление AdamW
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y):
        loss, gradients = loss_and_grad(model, x, y)
        optimizer.update(model, gradients)
        return loss

    # Вычисляем отложенные значения до начала замера
    mx.eval(state, train_x, train_y, val_x, val_y)

    for epoch in range(1, EPOCHS + 1):
        indices = mx.random.permutation(len(train_x))
        mx.eval(indices)

        started = time.perf_counter()
        last_loss = None

        batches = range(0, len(train_x), BATCH_SIZE)
        for start in tqdm(batches, desc=f"Epoch {epoch:02}"):
            ids = indices[start : start + BATCH_SIZE]
            last_loss = train_step(train_x[ids], train_y[ids])

            # Запускаем вычисление отложенного графа
            mx.eval(state, last_loss)

        mx.synchronize()
        elapsed = time.perf_counter() - started

        error = mx.array(0.0)

        for start in range(0, len(val_x), VAL_BATCH_SIZE):
            prediction = mx.mean(
                model(val_x[start : start + VAL_BATCH_SIZE]),
                axis=-1,
            )
            prediction = prediction * y_std + y_mean

            error += mx.sum(
                mx.abs(
                    prediction
                    - val_y[start : start + VAL_BATCH_SIZE]
                )
            )

        mx.eval(error)

        print(f">>> Epoch {epoch:02}")
        print(f"Last batch loss = {last_loss.item():.5f}")
        print(f"Validation MAE = {error.item() / len(val_x):.5f}")
        print(
            f"Training throughput = "
            f"{len(train_x) / elapsed:,.0f} objects/s"
        )
        print(">>>\n", flush=True)


if __name__ == "__main__":
    main()
