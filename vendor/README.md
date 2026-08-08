# Vendored veRL Runtime

`verl-safegrpo-0.7.0.dev.zip` is the runtime archive used by GRPO jobs. Its
observed SHA-256 is:

```text
883008dc84a2e8d9e797ae293f3d68b1ab69ed4c0052e669e64138ed8e6a1f58
```

Verify before use:

```bash
shasum -a 256 vendor/verl-safegrpo-0.7.0.dev.zip
```

`scripts/amlt/grpo/prepare_writable_verl.sh` extracts the archive into a writable runtime
directory, applies the repository's FSDP attention-loading patch, adds that
directory to `PYTHONPATH`, and verifies the import. GRPO AMLT jobs install
`requirements-verl.lock` before sourcing that script.

The archive filename suggests a veRL SafeGRPO development snapshot, but this
repository does not establish its upstream repository, source revision, build
procedure, modifications before vendoring, or license. Those facts are
**unknown** and must not be inferred from the filename. Confirm provenance and
license with the archive provider before redistribution or external use.
