# Test fixtures

Shared test data: sample manifests, reference images, recorded tracking output.

Most fixtures today are built programmatically in `tests/conftest.py`
(`make_manifest_dict`, `write_avatar`) rather than stored as files. That is
preferred where it works — a fixture built in code cannot drift out of sync with
the schema it is meant to exercise, because it is validated by the same
Pydantic models.

Use this directory for data that genuinely cannot be generated: a real captured
frame, a recorded tracker output to replay, a deliberately corrupted file whose
exact bytes matter.

Keep files small. Large fixtures make the repository heavy for everyone,
permanently.
