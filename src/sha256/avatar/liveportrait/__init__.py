"""LivePortrait's networks, vendored.

**Provenance.** Copied from https://github.com/KwaiVGI/LivePortrait at commit
``9b294b3d0536135442ea73cb01e6cb3ca7029dd3`` on 2026-09-04: the eight modules
under ``src/modules`` and ``src/utils/camera.py``, plus ``mask_template.png``
from ``src/utils/resources`` (kept for provenance; **unused** as of
2026-09-05 -- ``engine_liveportrait._oval_mask`` builds a mask from the
portrait's own landmarks instead, the way the warp engine's own composite
does, because the template takes no account of one portrait's proportions).
MIT, copyright (c) 2024 Kuaishou Visual
Generation and Interaction Center; the licence text is in ``LICENSE`` beside
this file. The weights are not here — they are fetched by
``scripts/fetch_liveportrait.py`` into ``models/liveportrait`` and never enter
Git. Nothing under ``insightface/`` is used anywhere: face finding is
MediaPipe's, Apache-2.0.

**What was changed, and why it is exactly this much.** Four tensors were built
on the CPU inside the forward pass and moved to the GPU — the coordinate grid
in ``util.make_coordinate_grid``, the zero heat-map slab in
``dense_motion.create_heatmap_representations``, and the index and one/zero
columns in ``camera.py``. Each is now allocated on the device it is used on.
That is a no-op numerically and it is the difference between the whole frame
capturing into one CUDA graph and not capturing at all: measured 2026-09-04 on
the RTX 5070 Ti, 35.5 ms a frame eager against 25.2 ms replayed, identical
output. Nothing else is touched, so a later upstream version can be dropped in
and the same four lines re-applied.
"""
