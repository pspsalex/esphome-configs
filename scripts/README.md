# esphome scripts

## imagen.py

Generates images and ESPHome configurations for those images, based on the `images.yaml` configuration file.

Supports generating images based on:
* a font and text (usually a single UTF-8 or UTF-16 character pointing to an MDI or MSR icon), with configurable color and
  size
* an SVG input file and given size
* an image file (expected: PNG) and given output size. The image is resized to fit the given size, while keeping the aspect
  ratio.

### Prerequisites

Install the required Python dependencies:

```bash
pip install pillow cairosvg pyyaml"
```

### Quick Start

1. **Generate the icons:**
   ```bash
   python3 scripts/imagen.py
   ```

### Configuration

See `images.yaml` for an example.

Output folders for image generation, folder structure and expected input file structure are hard-coded in the script and
assume the current repository layout.
