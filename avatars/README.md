# Avatar library

Each subdirectory is one avatar, named exactly by its manifest `id`.

```
avatars/
  avatar_001/
    metadata.json     <- required; the manifest
    preview.png       <- optional; shown in the gallery
    model/            <- weights and engine artifacts (NOT tracked in Git)
    config/           <- optional engine-specific extras
```

Adding an avatar is a directory copy. There is no central index to edit and no
application code to change — the library is scanned at startup and manifests
are validated individually, so one bad avatar cannot hide the others.

The schema lives in `src/sha256/core/avatar.py`; the full design is documented
in `docs/architecture/AVATAR_SYSTEM.md`.

## What is tracked in Git

| Path | Tracked | Why |
|---|---|---|
| `metadata.json` | yes | Small, reviewable, and defines the library layout |
| `preview.png` | no | Owner-supplied image data |
| `model/` | no | Large, and redistribution may be restricted by the model's license |

## Sample avatars

`avatar_001` and `avatar_002` are placeholders that reference no weights and
render through the null engine. They exist so discovery, validation, selection,
and residency eviction are exercised by tests rather than assumed. Delete them
once real avatars exist.

All avatars in sha256 are fictional and AI-generated. Manifests carry a
`fictional: true` assertion; see `docs/PRIVACY.md`.
