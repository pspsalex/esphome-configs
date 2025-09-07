# esphome scripts

## generate_icons.py

The script is used to generate image icons initially referenced as UTF characters from the material icons fonts. It
takes a list of font names and sizes as input and generates PNG files for each icon. Check the script for more details.

### Prerequisites

Install the required Python dependencies:

```bash
pip install requests pillow fonttools
```

### Quick Start

1. **Generate the icons:**
   ```bash
   cd esphome-configs
   python3 scripts/generate_icons.py
   ```


### Generation Options

#### Automatic Download (Default)

The script will automatically download the Material Symbols Rounded font from Google Fonts:

```bash
python3 scripts/generate_icons.py
```

#### Use Local Font

If you already have the font file or want to use a specific version:

```bash
# Place your font file as assets/fonts/MaterialSymbolsRounded.ttf
python3 scripts/generate_icons.py --use-local-font
```

#### Custom Colors

Generate icons in different colors:

```bash
python3 scripts/generate_icons.py --color white  # For dark backgrounds
python3 scripts/generate_icons.py --color gray   # For subtle appearance
```


### Troubleshooting

#### Download Failure

If the automatic download fails:

   - Go to [Google Fonts Material Symbols](https://fonts.google.com/icons)
   - Download the Material Symbols Rounded font
   - Extract and place the `.ttf` file as `assets/fonts/MaterialSymbolsRounded.ttf`
   - Run: `python3 scripts/generate_icons.py --use-local-font`

#### Missing Icons

If specific icons don't generate:
1. Check the Unicode values in `scripts/generate_icons.py`
2. Verify the font supports those characters
3. Use fallback generation for problematic icons

### Customization

#### Adding New Icons

1. Find the Unicode value for the Material Symbol
2. Add it to the `ICONS` dictionary in `scripts/generate_icons.py`
3. Re-run the generation script
4. Add the image definition to `config/icons.yaml`

#### Custom Sizes

Modify the `SIZES` dictionary in `scripts/generate_icons.py` to add new size variants.
