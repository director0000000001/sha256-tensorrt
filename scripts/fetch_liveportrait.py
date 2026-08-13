"""Download LivePortrait's model weights — and deliberately not InsightFace's.

**Why this script exists rather than a documented `git clone`.** LivePortrait's
own instructions fetch the whole `pretrained_weights` tree, which includes
InsightFace's `buffalo_l` detection models. Those models are **non-commercial
research only** — verbatim from the InsightFace repository — and LivePortrait's
documentation says so itself: commercial use requires removing and replacing
them.

sha256 does not need them. It already tracks faces with MediaPipe Face
Landmarker, which is Apache-2.0 on both code and weights (B003, and A1 in
``docs/OPEN_SOURCE_EVALUATION.md``), and MediaPipe's landmarks are what would
drive the crop anyway. So this fetches **only** the files under
``liveportrait/``, and never touches ``insightface/``.

That is not a detail. It is the difference between a build that can be sold and
one that cannot, and doing it by omission here is far safer than doing it by
deletion later.

**The licence position on what this does download**, verified 2026-08-14 from
the primary source: the model repository's own card declares ``license: mit``
in its metadata. See ``docs/OPEN_SOURCE_EVALUATION.md`` B1 for the full
statement, including what is still not confirmed.

Usage::

    python scripts/fetch_liveportrait.py
    python scripts/fetch_liveportrait.py --dest models/liveportrait
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REPO_ID = "KwaiVGI/LivePortrait"

# Only the generator stack. Everything here is LivePortrait's own work; nothing
# under insightface/ is requested, and that omission is the point.
WANTED = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
    "liveportrait/landmark.onnx",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=str(REPO_ROOT / "models" / "liveportrait"),
        help="Where to put the weights. Not in Git — they are hundreds of MB.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface-hub is not installed:  uv pip install huggingface-hub")
        return 2

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"repository : {REPO_ID}")
    print(f"destination: {dest}")
    print("skipping   : insightface/** (non-commercial research licence)")
    print()

    total = 0
    for name in WANTED:
        try:
            path = Path(hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(dest)))
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")
            return 1
        size = path.stat().st_size
        total += size
        print(f"  {name:<62} {size / 1e6:8.1f} MB")

    print()
    print(f"{len(WANTED)} file(s), {total / 1e6:.0f} MB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
