"""Train the refiner that corrects what the warp gets wrong, on CUDA.

**What it is.** A small convolutional network that takes the warp's own output
and predicts a *correction* to add to it. Not a face generator: the input is
already the right person in the right pose with the right expression, and the
only thing missing is that some of it is stretched, sheared, or invented out of
neighbouring skin. That is a far smaller problem than synthesising a face, and
B001 measured this shape at **1.83 ms fp32 / 1.33 ms fp16** against a 33.3 ms
frame budget -- which is the entire reason the plan has a refiner in it.

**Residual, not replacement.** The network outputs a delta and the result is
``warped + delta``. A network asked to output the picture has to learn to
reproduce a correct face before it can improve one, and every pixel the warp
already got right becomes something it can get wrong. Starting from zero means
an untrained refiner is the identity, so training can only be measured as an
improvement on the warp rather than as a recovery from a worse start.

**The split is by clip, and that is not a detail.** Frames five apart in one
clip are nearly the same photograph, so a random split puts near-duplicates on
both sides and reports a validation number that means nothing at all. A whole
clip is held out, so the reported number is the refiner meeting a take it has
never seen.

**The loss is masked and the metric is not the loss.** Outside the face the warp
has no opinion and the two images differ by the whole background, so the loss is
computed inside the mask only. And it is reported two ways: the mean over the
face, and the **worst 1% of it**, because a sheared triangle by an eye corner is
a handful of pixels wrong by a lot and moves the mean by almost nothing. The
owner's complaints are all worst-1% complaints.

Usage::

    uv run python scripts/train_refiner.py --pairs data/pairs-v1
    uv run python scripts/train_refiner.py --epochs 60 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Held out by default: a clip of its own, never trained on, so the number the
# run reports is the refiner meeting a take it has not seen.
DEFAULT_HOLDOUT = "video-10"

# Width of the network's first stage. Everything downstream is a multiple of it,
# so this is the one knob that trades quality against the frame budget.
DEFAULT_WIDTH = 32


def build_model(width: int, blocks: int):
    """A small encoder, a few residual blocks, a decoder, and a delta out.

    Two stride-2 convolutions down and two nearest-neighbour upsamples back,
    which is the shape B001 costed. Nearest-neighbour rather than a transposed
    convolution because transposed convolutions put a checkerboard on smooth
    skin, and smooth skin is most of a face.
    """
    import torch
    from torch import nn

    class Residual(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x + self.body(x))

    class Refiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            deep = width * 2
            self.down = nn.Sequential(
                nn.Conv2d(3, width, 3, stride=2, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, deep, 3, stride=2, padding=1),
                nn.GroupNorm(8, deep),
                nn.SiLU(inplace=True),
            )
            self.middle = nn.Sequential(*[Residual(deep) for _ in range(blocks)])
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(deep, width, 3, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(8, width),
                nn.SiLU(inplace=True),
            )
            self.delta = nn.Conv2d(width, 3, 3, padding=1)
            # Zero-initialised, so the network starts as exactly the identity
            # and the first epoch cannot be worse than the warp alone.
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

        def forward(self, warped: torch.Tensor) -> torch.Tensor:
            features = self.up(self.middle(self.down(warped)))
            return warped + self.delta(features)

    return Refiner()


def masked_scores(prediction, truth, mask) -> tuple[float, float]:
    """Mean absolute error over the face and over its worst 1%, in 0-255."""
    import torch

    error = (prediction - truth).abs().mean(dim=1) * 255.0
    inside = mask > 0.5
    if not bool(inside.any()):
        return float("nan"), float("nan")
    values = error[inside]
    mean = float(values.mean())
    worst = float(torch.quantile(values.float(), 0.99)) if values.numel() > 1 else mean
    return mean, worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPO_ROOT / "data" / "pairs-v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "refiner")
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp", action="store_true", help="Train in mixed precision.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Train on this share of the training pairs. For the learning curve: "
            "if held-out error falls as the share rises, the answer is more data."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Run scripts/check_gpu.py.")
        return 1
    device = torch.device(args.device)
    print(f"device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")

    blob = np.load(args.pairs / "pairs.npz")
    meta = json.loads((args.pairs / "index.json").read_text(encoding="utf-8"))
    clips = [p["clip"] for p in meta["pairs"]]
    if len(clips) != len(blob["warped"]):
        print("index.json and pairs.npz disagree about how many pairs there are.")
        return 1

    held = np.array([clip == args.holdout for clip in clips])
    if not held.any():
        print(f"No pairs come from {args.holdout!r}; nothing would be held out.")
        return 1
    print(
        f"{len(clips)} pairs, {int((~held).sum())} for training, "
        f"{int(held.sum())} held out ({args.holdout})"
    )

    def tensors(select: np.ndarray):
        warped = torch.from_numpy(blob["warped"][select]).permute(0, 3, 1, 2).float() / 255.0
        truth = torch.from_numpy(blob["truth"][select]).permute(0, 3, 1, 2).float() / 255.0
        mask = torch.from_numpy(blob["mask"][select]).float() / 255.0
        return warped, truth, mask

    training = ~held
    if args.train_fraction < 1.0:
        # Thinned evenly across the clips rather than by taking a prefix, so a
        # smaller share is a smaller sample of the same material and not a
        # sample of the first two clips.
        available = np.flatnonzero(training)
        keep = available[:: max(1, round(1.0 / args.train_fraction))]
        training = np.zeros_like(training)
        training[keep] = True
        print(f"training on {int(training.sum())} of {len(available)} (--train-fraction)")

    train_x, train_y, train_m = tensors(training)
    val_x, val_y, val_m = tensors(held)
    val_x, val_y, val_m = val_x.to(device), val_y.to(device), val_m.to(device)

    model = build_model(args.width, args.blocks).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"refiner: {parameters / 1000:.0f}k parameters, width {args.width}, {args.blocks} blocks")

    # The warp on its own, on the held-out clip. Every number below is measured
    # against this and not against zero -- the question is never "is the refiner
    # good", it is "is it better than not having one".
    with torch.no_grad():
        base_mean, base_worst = masked_scores(val_x, val_y, val_m)
    print(f"\nthe warp alone, held out:  mean {base_mean:.2f}   worst 1% {base_worst:.2f}\n")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    order = np.arange(len(train_x))
    generator = np.random.default_rng(0)

    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        generator.shuffle(order)
        total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch):
            picks = order[start : start + args.batch]
            x = train_x[picks].to(device, non_blocking=True)
            y = train_y[picks].to(device, non_blocking=True)
            m = train_m[picks].to(device, non_blocking=True).unsqueeze(1)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                prediction = model(x)
                # Masked L1: the mean is taken over the face's own pixels, not
                # over the tile, so a small face in a big box is not scored as
                # mostly-correct because the background matched.
                loss = ((prediction - y).abs() * m).sum() / (m.sum() * 3.0 + 1e-6)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            total += float(loss.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            predicted = torch.cat(
                [model(val_x[i : i + args.batch]) for i in range(0, len(val_x), args.batch)]
            )
            mean, worst = masked_scores(predicted.clamp(0.0, 1.0), val_y, val_m)

        marker = ""
        if mean < best:
            best = mean
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "width": args.width,
                    "blocks": args.blocks,
                    "holdout": args.holdout,
                    "val_mean": mean,
                    "val_worst": worst,
                },
                args.out / "refiner.pt",
            )
            marker = "  <- saved"

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(
                f"epoch {epoch:>3}  train {total / max(batches, 1) * 255.0:>6.2f}   "
                f"held out mean {mean:>6.2f} ({mean - base_mean:+.2f})   "
                f"worst 1% {worst:>6.2f} ({worst - base_worst:+.2f}){marker}"
            )

    print(f"\ntrained in {time.perf_counter() - started:.0f}s")

    # What it costs per frame, which is the number that decides whether any of
    # this ships. Measured on one tile, the way the engine would call it.
    model.eval()
    one = val_x[:1]
    with torch.no_grad():
        for _ in range(20):
            model(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            model(one)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    print(
        f"inference, one 256x256 tile: median {timings[len(timings) // 2]:.2f} ms   "
        f"p95 {timings[94]:.2f} ms"
    )
    print("animate is 21.4 ms; the frame budget at 30 fps is 33.3 ms")
    print(f"\nbest held-out mean {best:.2f} against the warp's {base_mean:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
