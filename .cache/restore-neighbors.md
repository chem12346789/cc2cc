# Restore the removed neighbor implementation

The archive `.cache/neighbors-before-removal.tar.gz` contains the exact
pre-removal contents of every affected source and documentation file,
including any changes that were already present.
The backup is untracked; cache cleanup or `git clean` can remove it.

Run these commands from the repository root:

```bash
printf '%s  %s\n' \
  '5f621b124305f61ef16ed125d6bea290a4d053aed255594a39607b858a74fe2d' \
  '.cache/neighbors-before-removal.tar.gz' | sha256sum -c - &&
tar -tzf .cache/neighbors-before-removal.tar.gz &&
tar -xzf .cache/neighbors-before-removal.tar.gz
```

Extracting the archive overwrites all listed files. Save any later edits to
those files first; unrelated files are not changed. To restore only selected
files, append their paths from the listing to the extraction command, for
example:

```bash
tar -xzf .cache/neighbors-before-removal.tar.gz cc2cc/utils/ModelClass.py
```
