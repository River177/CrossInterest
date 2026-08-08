# Slides

`build_novel_interest_deck.py` generates the novel-interest presentation from
repository content and `assets/novel_interest_prompt.png`.

```bash
python slides/build_novel_interest_deck.py
```

The builder requires `python-pptx`, which is not included in the local package
dependencies. It writes
`slides/output/generative_novel_interest_prediction.pptx`. Rendered PNG/PDF
previews in `slides/output/` are generated artifacts and may require separate
presentation rendering tools; the entire output directory is ignored by Git.

Treat the Python builder and `slides/assets/` as source. Restore or regenerate
ignored outputs as described in [manifests/README.md](../manifests/README.md).
