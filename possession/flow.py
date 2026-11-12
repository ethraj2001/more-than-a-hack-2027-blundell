"""LSTM flow matching: a second model beside the mixture of experts, from the same features.

A BiLSTM encodes the features (features.compute, with Smart Tagging's series when the model reads
them); a second BiLSTM, the denoiser, is trained by discrete flow matching: every training
window's labels are masked in blocks, each with probability 1 - t for t drawn uniformly, and the
masked labels predicted from the encoded features and the labels left showing. So it learns what
a possession looks like next to the ones around it as well as what the tracking says.

What the pipeline reads is the flow's first step: P(H, A, D) per frame with every label still
masked (first_round), blended with the gate's before the decoder (config.FLOW_WEIGHT). Running the
flow on -- unmasking over SAMPLE_STEPS rounds, most confident first, each round reading the labels
already fixed (sample) -- was measured and scored lower than its first step; see config.

Team symmetry as in the experts: the features are read in both team orders, and the second
order's output is swapped back and averaged in. The second order is the first with its A and B
columns exchanged and A-minus-B negated (swap_columns), so only one is stored.

Training is seeded but runs on whatever device it is given, and a GPU's floats are not
reproducible; prediction runs on the CPU in float64 and its probabilities are rounded to ROUND
decimals, so the same weights give the same file on any machine.
"""

import numpy as np
import torch
from torch import nn

from possession.scoring import A, D, H, X

# Params
WINDOW = 1500              # frames per training window and per prediction chunk (60 s)
STRIDE = 750               # prediction chunks overlap by half; each frame read from the nearest centre
BATCH = 32
TRAIN_STEPS = 3000
LEARNING_RATE = 1e-3
WARMUP = 100
WEIGHT_DECAY = 1e-2
HIDDEN = 128
EMBED = 16
DROPOUT = 0.1
BLOCKS = (5, 25, 125, 625)  # mask block lengths (frames), one drawn per window
FULL_MASK = 0.0            # share of training windows with every label masked: the first step's case;
                           # 0.5 measured: blend +0.004 [-0.000, +0.008], flow alone -0.009 (README)
SAMPLE_STEPS = 8
CLIP = 5.0                 # standardised features are clipped to +-CLIP
ROUND = 6                  # decimals the predicted probabilities are rounded to
MASK = 3                   # the masked label; H, A, D are 0, 1, 2
SWAP = np.array([A, H, D, MASK])  # labels under the other team order


class FlowNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(2 * n_features, HIDDEN),
                                 nn.GELU())
        self.encoder = nn.LSTM(HIDDEN, HIDDEN, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=DROPOUT)
        self.embed = nn.Embedding(4, EMBED)
        self.denoiser = nn.LSTM(2 * HIDDEN + EMBED, HIDDEN, num_layers=1, bidirectional=True,
                                batch_first=True)
        self.out = nn.Linear(2 * HIDDEN, 3)

    def forward(self, x, y):
        """x: (B, L, 2F) encoded features; y: (B, L) labels with MASK. Logits (B, L, 3)."""
        context, _ = self.encoder(self.inp(x))
        hidden, _ = self.denoiser(torch.cat([context, self.embed(y)], dim=-1))
        return self.out(hidden)


def swap_columns(names):
    """(index, sign): matrix[:, index] * sign is the matrix in the other team order."""
    position = {name: j for j, name in enumerate(names)}
    index, sign = [], []
    for name in names:
        prefix, key = name.split(".", 1)
        other = {"A": "B." + key, "B": "A." + key}.get(prefix, name)
        index.append(position[other])
        sign.append(-1.0 if prefix == "AmB" else 1.0)
    return np.array(index), np.array(sign, dtype=np.float16)


def scaler(matrices, rng, sample=200000):
    """(mean, sd) per feature over a sample of frames of the training matrices, NaN ignored."""
    rows = [m[rng.integers(len(m), size=max(1, sample // len(matrices)))] for m in matrices]
    stacked = np.concatenate(rows).astype(np.float32)
    mean = np.nan_to_num(np.nanmean(stacked, axis=0))
    sd = np.nan_to_num(np.nanstd(stacked, axis=0))
    return mean.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)


def encode(matrix, scale):
    """(..., F) features with NaN -> (..., 2F) float16: standardised, clipped, NaN as 0, then a
    seen flag per feature, so the model can tell a missing value from an average one."""
    mean, sd = scale
    values = (matrix.astype(np.float32) - mean) / sd
    seen = ~np.isnan(values)
    values = np.clip(np.nan_to_num(values), -CLIP, CLIP)
    return np.concatenate([values, seen], axis=-1).astype(np.float16)


def _masked(labels, valid, rng):
    """Training input for one window: labels masked in blocks with probability 1 - t, or, for a
    FULL_MASK share of windows, every label masked."""
    if FULL_MASK and rng.random() < FULL_MASK:
        return np.full(len(labels), MASK, dtype=labels.dtype), valid.copy()
    t = rng.random()
    block = BLOCKS[rng.integers(len(BLOCKS))]
    hidden = rng.random(len(labels) // block + 1) < 1.0 - t
    hidden = np.repeat(hidden, block)[:len(labels)] | ~valid
    return np.where(hidden, MASK, labels), hidden & valid


def train(sequences, swap, scale, device="cpu", steps=TRAIN_STEPS, seed=0, log=print):
    """sequences: [{"x": (T, F) float16 features, "y": (T,) labels, "valid": (T,) bool}] of the
    training copies, in one team order; swap from swap_columns(). Returns the fitted FlowNet, on
    the CPU, in eval mode."""
    index, sign = swap
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = FlowNet(len(index)).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda k: min(1.0, (k + 1) / WARMUP)
                                                 * 0.5 * (1 + np.cos(np.pi * k / steps)))
    lengths = np.array([len(s["y"]) - WINDOW for s in sequences], dtype=np.float64)
    weights = lengths / lengths.sum()
    net.train()
    running = []
    for step in range(steps):
        xs, ys, targets = [], [], []
        for which in rng.choice(len(sequences), BATCH, p=weights):
            s = sequences[which]
            start = int(rng.integers(len(s["y"]) - WINDOW))
            y, valid = s["y"][start:start + WINDOW], s["valid"][start:start + WINDOW]
            x = s["x"][start:start + WINDOW]
            if rng.random() < 0.5:
                x, y = x[:, index] * sign, SWAP[y]
            y_in, scored = _masked(y, valid, rng)
            xs.append(encode(x, scale)), ys.append(y_in)
            targets.append(np.where(scored, y, -100))
        x = torch.from_numpy(np.stack(xs)).to(device).float()
        y_in = torch.from_numpy(np.stack(ys).astype(np.int64)).to(device)
        target = torch.from_numpy(np.stack(targets).astype(np.int64)).to(device)
        logits = net(x, y_in)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 3), target.reshape(-1),
                                           ignore_index=-100)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        running.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            log("   flow step {} loss {:.4f}".format(step + 1, np.mean(running)))
            running = []
    return net.to("cpu").eval()


def _chunks(n):
    starts = list(range(0, max(1, n - WINDOW + 1), STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    centres = np.array(starts) + WINDOW / 2.0
    nearest = np.searchsorted((centres[1:] + centres[:-1]) / 2.0, np.arange(n), side="right")
    return starts, nearest


@torch.no_grad()
def proba(net, x, y, swap, scale, device="cpu", dtype=torch.float32, batch=BATCH):
    """(T, 3) P(H, A, D) per frame of features x (T, F) given labels y (MASK where open): both
    team orders, chunked."""
    index, sign = swap
    n = len(y)
    starts, nearest = _chunks(n)
    net = net.to(device=device, dtype=dtype)
    out = np.zeros((n, 3), dtype=np.float64)
    for swapped in (False, True):
        chunks = []
        for b in range(0, len(starts), batch):
            idx = starts[b:b + batch]
            raw = np.stack([x[s:s + WINDOW] for s in idx])
            labels = np.stack([y[s:s + WINDOW] for s in idx])
            if swapped:
                raw, labels = raw[..., index] * sign, SWAP[labels]
            xx = torch.from_numpy(encode(raw, scale)).to(device=device, dtype=dtype)
            yy = torch.from_numpy(labels.astype(np.int64)).to(device)
            chunks.append(torch.softmax(net(xx, yy), dim=-1).cpu().double().numpy())
        chunks = np.concatenate(chunks)
        p = chunks[nearest, np.arange(n) - np.array(starts)[nearest]]
        out += p[:, [1, 0, 2]] if swapped else p
    return out / 2.0


def first_round(net, x, swap, scale):
    """(T, 3) P(H, A, D): the flow's first step, every label masked -- on the CPU in float64,
    rounded to ROUND decimals, so the same weights give the same numbers anywhere."""
    p = proba(net, x, np.full(len(x), MASK), swap, scale, "cpu", torch.float64)
    return np.round(p, ROUND)


def sample(net, x, period, swap, scale, device="cpu", steps=SAMPLE_STEPS):
    """Label codes (T,) for one match, and the first round's P(H, A, D): every label masked,
    unmasked over steps rounds, most confident first; X outside play."""
    valid = period > 0
    y = np.full(len(period), MASK, dtype=np.int64)
    first = None
    total = int(valid.sum())
    for k in range(steps):
        p = proba(net, x, y, swap, scale, device)
        if first is None:
            first = p
        open_ = np.flatnonzero((y == MASK) & valid)
        n_fix = len(open_) - int(round(total * (1.0 - (k + 1) / steps)))
        if n_fix <= 0:
            continue
        fix = open_[np.argsort(-p[open_].max(axis=1), kind="stable")[:n_fix]]
        y[fix] = p[fix].argmax(axis=1)
    return np.where(valid, y, X).astype(np.int8), first


def state(net):
    """The weights as numpy arrays, to pickle in a bundle."""
    return {key: value.detach().cpu().numpy() for key, value in net.state_dict().items()}


def restore(weights, n_features):
    net = FlowNet(n_features)
    net.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()})
    return net.eval()
