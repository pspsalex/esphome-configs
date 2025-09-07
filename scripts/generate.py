#!/usr/bin/env python3
"""
SPDX-License-Identifier: MIT
Copyright (c) 2025 Alex Popescu (@pspsalex)
AI generated

Modern Material Symbols icon generator for ESPHome LVGL configurations.

This script downloads the Material Symbols Rounded font and generates
PNG images for all defined icons with proper error handling, type safety,
and extensibility.

Dependencies:
    pip install requests pillow fonttools

Usage:
    python3 scripts/generate_icons.py
    python3 scripts/generate_icons.py --use-local-font
"""

from __future__ import annotations

from icon_definitions import IconDefinitions, Icon, IconSize

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

try:
    import requests
    from PIL import Image, ImageDraw, ImageFont
    import yaml
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install requests pillow fonttools")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
SCRIPT_DIR = Path(__file__).parent
ASSETS_BASE = SCRIPT_DIR.parent.parent
GOOGLE_FONTS_API = "https://fonts.googleapis.com/css2"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class FontFormat(Enum):
    """Supported font formats."""
    TTF = "ttf"
    WOFF2 = "woff2"
    OTF = "otf"


class Color(Enum):
    """Available icon colors."""
    BLACK = (0, 0, 0, 255)
    WHITE = (255, 255, 255, 255)
    GRAY = (128, 128, 128, 255)


@dataclass
class FontDownloadResult:
    """Result of font download operation."""
    success: bool
    path: Optional[Path] = None
    format: Optional[FontFormat] = None
    error: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for icon generation."""
    use_local_font: bool = False
    font_name: str = "MaterialSymbolsRounded"
    color: Color = Color.BLACK
    output_base: Path = field(default_factory=lambda: ASSETS_BASE)
    theme: str = "default"

    @property
    def fonts_dir(self) -> Path:
        """Get fonts directory path."""
        return self.output_base / "fonts"

    @property
    def icons_dir(self) -> Path:
        """Get icons directory path."""
        return self.output_base / "themes" / self.theme / "icons"





class DependencyChecker:
    """Check and validate required dependencies."""

    REQUIRED_PACKAGES = ["requests", "pillow", "pyyaml"]
    OPTIONAL_PACKAGES = {"fonttools": "woff2 conversion"}

    @classmethod
    def check_required(cls) -> bool:
        """Check if all required dependencies are available."""
        missing = []

        for package in cls.REQUIRED_PACKAGES:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)

        if missing:
            logger.error(f"Missing required dependencies: {', '.join(missing)}")
            logger.info(f"Install with: pip install {' '.join(missing)}")
            return False

        return True

    @classmethod
    def check_optional(cls) -> Dict[str, bool]:
        """Check availability of optional dependencies."""
        available = {}

        for package, purpose in cls.OPTIONAL_PACKAGES.items():
            try:
                __import__(package)
                available[package] = True
                logger.debug(f"Optional dependency {package} available ({purpose})")
            except ImportError:
                available[package] = False
                logger.debug(f"Optional dependency {package} not available ({purpose})")

        return available


class FontManager:
    """Manage font download, conversion, and local discovery."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._font_url_patterns = [
            (FontFormat.TTF, r'url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)'),
        ]

    def find_local_font(self) -> Optional[Path]:
        """Find existing local font file."""
        possible_paths = [
            self.config.output_base / "themes" / self.config.theme / "fonts" / f"{self.config.font_name}.{fmt.value}"
            for fmt in [FontFormat.TTF]
        ]

        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"Found local font: {path}")
                return path

        return None

    def download_font(self) -> FontDownloadResult:
        """Download Material Symbols font from Google Fonts."""
        logger.info("Downloading Material Symbols Rounded font...")

        try:
            # Get CSS with font URLs
            api_url = f"{GOOGLE_FONTS_API}?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0"
            response = requests.get(api_url, headers={'User-Agent': USER_AGENT}, timeout=30)
            response.raise_for_status()

            css_content = response.text
            font_info = self._extract_font_url(css_content)

            if not font_info:
                return FontDownloadResult(
                    success=False,
                    error="No supported font format found in CSS response"
                )

            font_format, font_url = font_info
            logger.info(f"Found {font_format.value.upper()} font URL: {font_url}")

            # Download font file
            font_response = requests.get(font_url, timeout=60)
            font_response.raise_for_status()

            # Save and optionally convert
            font_dir = self.config.output_base / "themes" / self.config.theme / "fonts"
            font_dir.mkdir(parents=True, exist_ok=True)

            if font_format == FontFormat.TTF:
                return self._save_ttf_font(font_response.content)

        except requests.RequestException as e:
            return FontDownloadResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            return FontDownloadResult(success=False, error=f"Unexpected error: {e}")

    def _extract_font_url(self, css_content: str) -> Optional[Tuple[FontFormat, str]]:
        """Extract font URL and format from CSS content."""
        for font_format, pattern in self._font_url_patterns:
            match = re.search(pattern, css_content)
            if match:
                return font_format, match.group(1)
        return None

    def _save_ttf_font(self, font_data: bytes) -> FontDownloadResult:
        """Save TTF font data directly."""
        font_path = self.config.output_base / "themes" / self.config.theme / "fonts" / f"{self.config.font_name}.ttf"

        try:
            font_path.write_bytes(font_data)
            logger.info(f"Saved TTF font to: {font_path}")
            return FontDownloadResult(success=True, path=font_path, format=FontFormat.TTF)
        except Exception as e:
            return FontDownloadResult(success=False, error=f"Failed to save TTF: {e}")


class IconGenerator:
    """Generate icon images from font."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.icons = {icon.name: icon for icon in IconDefinitions.ICONS}
        self.sizes = {size.name: size for size in IconDefinitions.SIZES}

    def generate_font_icon(
        self,
        font_path: Path,
        icon: Icon,
        size: IconSize,
        output_path: Path
    ) -> bool:
        """Generate a single icon image using font."""
        try:
            # Create transparent image
            img = Image.new('RGBA', (size.pixels, size.pixels), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # Load font with fallback sizes
            font = self._load_font_with_fallback(font_path, size.pixels)
            if not font:
                logger.error(f"Could not load font for {icon.name} at {size.pixels}px")
                return False

            # Calculate centered position
            x = size.pixels // 2
            y = size.pixels // 2

            # Draw icon
            draw.text((x, y), icon.unicode_char, font=font, fill=self.config.color.value, anchor="mm")

            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, "PNG", optimize=True)

            return True

        except Exception as e:
            logger.error(f"Error generating icon {icon.name}: {e}")
            return False

    def _load_font_with_fallback(self, font_path: Path, target_size: int) -> Optional[ImageFont.FreeTypeFont]:
        """Load font with fallback sizes if target size fails."""
        fallback_sizes = [
            target_size,
            target_size - 4,
            target_size - 8,
            int(target_size * 0.9),
            int(target_size * 0.8)
        ]

        for font_size in fallback_sizes:
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except (OSError, IOError):
                continue

        return None


class IconGeneratorApp:
    """Main application class orchestrating the icon generation process."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.font_manager = FontManager(config)
        self.icon_generator = IconGenerator(config)

    def run(self) -> int:
        """Execute the icon generation process."""
        logger.info("Material Symbols Icon Generator for ESPHome")
        logger.info("=" * 50)

        # Get font
        font_path = self._get_font_path()
        if not font_path:
            logger.error("Could not obtain font.")
            return 1

        # Generate icons
        return self._generate_font_icons(font_path)

    def _get_font_path(self) -> Optional[Path]:
        """Get font path from local files or download."""
        if self.config.use_local_font:
            font_path = self.font_manager.find_local_font()
            if not font_path:
                logger.error("No local font found. Available options:")
                logger.info(f"1. Download {self.config.font_name} font manually")
                logger.info(f"2. Place it as assets/gen/fonts/{self.config.font_name}.ttf")
                logger.info("3. Or run without --use-local-font to auto-download")
            return font_path
        else:
            # Try local first, then download
            font_path = self.font_manager.find_local_font()
            if font_path:
                return font_path

            result = self.font_manager.download_font()
            if result.success:
                return result.path
            else:
                logger.error(f"Font download failed: {result.error}")
                return None

    def _generate_font_icons(self, font_path: Path) -> int:
        """Generate icons using the font."""
        logger.info(f"Using font: {font_path}")
        logger.info(f"Generating icons in: {self.config.icons_dir}")
        logger.info(f"Icon color: {self.config.color.name.lower()}")

        # Create output directories
        self.config.icons_dir.mkdir(parents=True, exist_ok=True)
        for size in IconDefinitions.SIZES:
            (self.config.icons_dir / size.name).mkdir(exist_ok=True)

        # Generate icons
        total_icons = len(IconDefinitions.ICONS) * len(IconDefinitions.SIZES)
        generated = 0

        for icon in IconDefinitions.ICONS:
            for size in IconDefinitions.SIZES:
                output_path = self.config.icons_dir / size.name / f"{icon.name}.png"

                if self.icon_generator.generate_font_icon(font_path, icon, size, output_path):
                    generated += 1
                    logger.debug(f"Generated {icon.name} ({size.name}: {size.pixels}px)")
                else:
                    logger.warning(f"Failed to generate {icon.name} ({size.name}: {size.pixels}px)")

        self._finish_generation(generated, total_icons)
        self._generate_icons_yaml()
        return 0

    def _finish_generation(self, generated: int, total: int) -> None:
        """Complete the generation process with documentation."""
        logger.info(f"Generated {generated}/{total} icons successfully!")
        logger.info("Icon generation complete!")

    def _generate_icons_yaml(self) -> None:
        """Generate the icons.yaml file for the theme."""
        logger.info(f"Generating icons.yaml for theme: {self.config.theme}")

        icons_yaml = []
        for icon in IconDefinitions.ICONS:
            for size in IconDefinitions.SIZES:
                icon_path = self.config.icons_dir / size.name / f"{icon.name}.png"
                icons_yaml.append({
                    "file": str(icon_path.relative_to(self.config.output_base)),
                    "id": f"icon_{icon.name}_{size.name}",
                    "resize": f"{size.pixels}x{size.pixels}",
                    "type": "RGB565",
                    "transparency": "alpha_channel",
                })

        output_path = self.config.output_base / "themes" / self.config.theme / "icons.yaml"
        with open(output_path, "w") as f:
            yaml.dump(icons_yaml, f, default_flow_style=False)

        logger.info(f"Generated {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate Material Symbols icons for ESPHome LVGL configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Download font and generate all icons
  %(prog)s --use-local-font    # Use existing local font file
  %(prog)s --font-name MyFont  # Use a custom local font
  %(prog)s --color gray        # Generate icons in gray color
        """
    )

    parser.add_argument(
        '--use-local-font',
        action='store_true',
        help='Use existing local font file instead of downloading from Google Fonts'
    )

    parser.add_argument(
        '--font-name',
        type=str,
        default='MaterialSymbolsRounded',
        help='Name of the local font file to use (without extension)'
    )

    parser.add_argument(
        '--color',
        type=str,
        choices=[color.name.lower() for color in Color],
        default='black',
        help='Icon color (default: black)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--theme',
        type=str,
        default='default',
        help='Name of the theme to generate icons for'
    )

    parser.add_argument(
        '--output-base',
        type=Path,
        default=ASSETS_BASE,
        help='Output directory for generated assets (default: assets)'
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    try:
        color = Color[args.color.upper()]
    except KeyError:
        logger.error(f"Invalid color: {args.color}")
        return 1

    config = GenerationConfig(
        use_local_font=args.use_local_font,
        font_name=args.font_name,
        color=color,
        output_base=args.output_base,
        theme=args.theme
    )

    # Run application
    app = IconGeneratorApp(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
