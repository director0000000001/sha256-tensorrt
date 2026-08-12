# Models

Shared model artifacts — weights, converted ONNX graphs, built TensorRT engines.

**Nothing here is tracked in Git.** Model files are large, and several candidate
models carry redistribution terms that would make committing them a license
violation rather than merely a bad idea. See `docs/OPEN_SOURCE_EVALUATION.md`.

```
models/
  downloads/     as-acquired files, unmodified
  checkpoints/   converted or optimised artifacts
  metadata/      manifests describing what is here (tracked)
```

`metadata/` is the exception: small JSON records of what was downloaded, from
where, at what version, and with what checksum. Those are worth reviewing and
worth keeping in history — they are how a build becomes reproducible without
committing gigabytes.

The application performs **no downloads** unless `models.allow_downloads` is
explicitly enabled. It is `false` by default (`docs/PRIVACY.md`). Model files are
fetched by explicit developer action, through a script, never by the running
application.

## What lives here

| Artifact | Fetch with | Licence |
|---|---|---|
| `downloads/face_landmarker.task` | `python scripts/fetch_face_landmarker.py` | Apache-2.0 |

`scripts/fetch_face_landmarker.py --verify` re-checks the local file against the
recorded checksum.
