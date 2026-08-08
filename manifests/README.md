# Restoration Manifests

Git ignores `data/`, `artifacts/`, and generated `slides/output/`. This directory
contains lightweight templates for recording how a local checkout restored
those files without committing the files themselves or fabricating storage
locations.

## Templates

- [`data.example.yaml`](data.example.yaml): raw, processed, or training data.
- [`artifacts.example.yaml`](artifacts.example.yaml): model, evaluation, W&B, or
  slide outputs.

Copy a template to a descriptive tracked or private record only when a real
restoration source is known. Use repository-relative destination paths. Leave
unknown optional fields blank or mark them `unknown`; never guess a URI,
checksum, license, owner, or access policy.

Do not record credentials, access tokens, signed URLs, private keys, or secrets.
Prefer durable source identifiers available to authorized collaborators. Verify
downloaded files against provider-supplied SHA-256 values, then record the
verification date and result.

These manifests are documentation, not executable download specifications.
Current repository history does not identify canonical restoration URIs.
