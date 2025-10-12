# Source Repository

* URL: [gh:KDE/oxygen-icons](https://github.com/KDE/oxygen-icons)
* Commit: [3ef0653](https://github.com/KDE/oxygen-icons/commit/3ef0653553954aa45ca666158908fe88cda9aa41)

# Alterations / Changes
SVGZ icons have been decompressed to plain text SVG format:
```bash
for f in *.svgz; do zcat "$f" > "${f%.svgz}.svg"; done
```
