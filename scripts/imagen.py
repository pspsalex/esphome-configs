#!/usr/bin/env python3
"""
SPDX-License-Identifier: MIT
Copyright (c) 2026 Alex Popescu (@pspsalex)

Modern Material Symbols image generator for ESPHome LVGL configurations.

This script generates PNG images for multiple themes based on a centralized
configuration. It supports generating images from both fonts and SVG files.
This version uses a clean architecture with proper separation of concerns.

Dependencies:
    pip install pillow cairosvg pyyaml

Usage:
    python3 scripts/imagen.py
"""

import argparse
import concurrent.futures
import logging
import logging.handlers
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
    import cairosvg
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install pillow cairosvg pyyaml")
    sys.exit(1)


# ============================================================================
# Path Configuration - Centralized path management
# ============================================================================


class PathConfig:
    """Centralized path configuration."""

    def __init__(self, script_path: Path):
        self.script_dir = script_path.parent
        self.project_dir = self.script_dir.parent
        self.assets_base = self.project_dir / "assets" / "gen"
        self.config_base = self.project_dir / "config"
        self.yaml_base = Path("assets") / "gen"
        self.yaml_theme_base = self.config_base / "lvgl" / "themes"
        self.image_config_path = self.script_dir / "images.yaml"

    def resolve_project_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the project directory."""
        return self.project_dir / path

    def get_theme_output_dir(self, theme_name: str) -> Path:
        """Get the output directory for a theme."""
        return self.assets_base / theme_name

    def get_image_output_path(self, theme_name: str, image_name: str, size_slug: str) -> Path:
        """Get the full output path for an image."""
        return self.get_theme_output_dir(theme_name) / size_slug / f"{image_name}.png"

    def get_yaml_output_path(self, theme_name: str) -> Path:
        """Get the YAML configuration output path for a theme."""
        return self.yaml_theme_base / theme_name / "images.gen.yaml"


# ============================================================================
# Data Classes - Configuration structure
# ============================================================================


@dataclass
class Font:
    name: str = ""
    path: str = ""
    weight: int = 400


@dataclass
class Text:
    font: Font = field(default_factory=Font)
    data: str = ""


@dataclass
class Size:
    color: str | None = None
    text: Text | None = None
    slug: str | None = None
    width: int | None = None
    height: int | None = None
    source: str | None = None
    skip: bool | None = None


@dataclass
class ImageConfig:
    name: str
    description: str = ""
    text: dict[str, Text] = field(default_factory=dict)
    source: str = ""
    color: str = ""
    sizes: dict[str, Size] = field(default_factory=dict)


@dataclass
class Theme:
    name: str
    description: str = ""
    color: str = ""
    generate: bool = True
    fonts: dict[str, Font] = field(default_factory=dict)
    sizes: dict[str, Size] = field(default_factory=dict)
    images: dict[str, ImageConfig] = field(default_factory=dict)


# ============================================================================
# Config Parser - Keep the original Config class intact
# ============================================================================


class Config:
    def __init__(self, raw_config):
        self.themes = {}

        self.config = raw_config

        raw_themes = raw_config.get("themes", {})
        for theme_name, theme_data in raw_themes.items():
            if theme_name not in self.themes:
                self.themes[theme_name] = self._parse_theme(theme_name, theme_data)

        for theme_name, theme_object in self.themes.items():
            for font_name, font_object in theme_object.fonts.items():
                path = Path(__file__).parent.parent / font_object.path
                if not path.exists():
                    raise FileNotFoundError(f"Font file for {theme_name}.fonts.{font_name} not found: {path}")

            for image_name, image_object in theme_object.images.items():
                for size_name, size_object in image_object.sizes.items():
                    size_object.slug = size_object.slug if size_object.slug else theme_object.sizes[size_name].slug
                    size_object.width = size_object.width if size_object.width else theme_object.sizes[size_name].width
                    size_object.height = size_object.height if size_object.height else theme_object.sizes[size_name].height
                    size_object.text = size_object.text if size_object.text else (image_object.text if not size_object.source else None)
                    size_object.source = size_object.source if size_object.source else (image_object.source if not size_object.text else None)
                    size_object.color = size_object.color if size_object.color else (
                        image_object.color if image_object.color else (
                            theme_object.sizes[size_name].color if theme_object.sizes[size_name].color else theme_object.color
                        )
                    )

                    if not size_object.slug:
                        raise ValueError(f"Slug not found for size '{theme_name}.images.{image_name}.sizes.{size_name}'")

                    if not size_object.width:
                        raise ValueError(f"Width not found for size '{theme_name}.images.{image_name}.sizes.{size_name}'")

                    if not size_object.height:
                        raise ValueError(f"Height not found for size '{theme_name}.images.{image_name}.sizes.{size_name}'")

                    if not size_object.text and not size_object.source:
                        raise ValueError(f"Text or source not found for size '{theme_name}.images.{image_name}.sizes.{size_name}'")

                    if size_object.text and not size_object.color:
                        raise ValueError(f"Color not found for text size '{theme_name}.images.{image_name}.sizes.{size_name}'")

                    if size_object.text and size_object.text.font and size_object.text.font not in theme_object.fonts:
                        raise ValueError(f"Font '{size_object.text.font}' not found for text size '{theme_name}.images.{image_name}.sizes.{size_name}'")

    def _parse_theme(self, theme_name, data: dict[str, Any]):
        from copy import deepcopy
        import re

        base = data.get("base", None)
        theme = Theme(name=theme_name)

        if base:
            if base not in self.themes:
                if base not in self.config["themes"]:
                    raise ValueError(f"Configuration for base theme '{base}' not found")

                self.themes[base] = None
                self.themes[base] = self._parse_theme(base, self.config["themes"][base])

            if not self.themes[base]:
                raise ValueError(
                    f"Cycle detected. Please check '{theme_name}' and '{base}'"
                )

            theme = deepcopy(self.themes[base])
            theme.generate = True

        theme.description = data.get("description", theme.description)
        theme.color = data.get("color", theme.color)
        theme.generate = data.get("generate", theme.generate)

        if "fonts" in data:
            for font_name, font_data in data["fonts"].items():
                theme.fonts[font_name] = (
                    theme.fonts[font_name] if font_name in theme.fonts else Font()
                )
                theme.fonts[font_name].name = font_data.get(
                    "name", theme.fonts[font_name].name
                )
                theme.fonts[font_name].path = font_data.get(
                    "path", theme.fonts[font_name].path
                )
                theme.fonts[font_name].weight = font_data.get(
                    "weight", theme.fonts[font_name].weight
                )

        if "sizes" in data:
            for size_name, size_data in data["sizes"].items():
                theme.sizes[size_name] = self._parse_size(size_name, size_data, theme.sizes)

        if "images" in data:
            for image_data in data["images"]:
                image_name = image_data["name"]

                image = (
                    theme.images[image_name]
                    if image_name in theme.images
                    else ImageConfig(image_name)
                )

                image.description = image_data.get("description", image.description)

                if "text" in image_data:
                    if "source" in image_data:
                        raise ValueError(f"Use either text or source in the config - check {theme_name}.images.{image_name}")
                    image.text = self._parse_text(image_data["text"], image.text)
                    image.source = None

                if "source" in image_data:
                    image.source = image_data.get("source", image.source)
                    image.text = None

                if "sizes" in image_data:
                    # must be an array
                    # each entry is either a name = reference to upstream
                    # or reference + override
                    for size_data in image_data["sizes"]:
                        if isinstance(size_data, dict):
                            size_ref = next(iter(size_data))
                            size_data = size_data[size_ref]
                        else:
                            size_ref = size_data
                            size_data = {}

                        size = self._parse_size(size_ref, size_data, image.sizes)

                        image.sizes[size_ref] = size

                theme.images[image_name] = image

        return theme

    def _parse_size(self, size_name: str, size_data: dict, container: dict) -> Size:
        output_size = container[size_name] if size_name in container else Size()

        # Don't inherit skips... for now at least.
        output_size.skip = False

        for attribute in ["width", "height", "color", "slug", "skip"]:
            if attribute in size_data:
                output_size.__setattr__(attribute, size_data[attribute])

        if "text" in size_data:
            if "source" in size_data:
                raise ValueError(f"Both text and source specified for size {size_name}")
            output_size.text = self._parse_text(size_data["text"], output_size.text)
            output_size.source = None

        if "source" in size_data:
            output_size.text = None
            output_size.source = size_data["source"]

        return output_size

    def _parse_text(self, text: str | dict, base: Text | None) -> Text:
        import re

        if not isinstance(text, dict):
            textre = re.compile("([a-zA-Z0-9-_]+):(.*)")
            matches = textre.match(text)
            if not matches:
                text = {
                    "data": text
                }
            else:
                text = {
                    "font": matches.group(1),
                    "data": matches.group(2),
                }

        output_text = base if base else Text()
        if "data" in text:
            output_text.data = text["data"]
        if "font" in text:
            output_text.font = text["font"]

        return output_text


# ============================================================================
# Image Generation - Separate rendering logic
# ============================================================================


class ImageRenderer:
    """Handles the actual image rendering operations."""

    def __init__(self, path_config: PathConfig):
        self.path_config = path_config

    def render_font_image(self, size: Size, font: Font) -> Image.Image | None:
        """Render an image from font glyphs."""
        try:
            font_path = self.path_config.resolve_project_path(font.path)
            img = Image.new("RGBA", (size.width, size.height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            pil_font = self._load_font_with_fallback(font_path, size.width)
            if not pil_font:
                logging.getLogger(__name__).error(f"Could not load font {font_path} at {size.width}px")
                return None

            draw.text(
                (size.width // 2, size.height // 2),
                size.text.data,
                font=pil_font,
                fill=ImageColor.getrgb(size.color),
                anchor="mm",
            )
            return img
        except Exception as e:
            logging.getLogger(__name__).error(f"Error rendering font image: {e}")
            return None

    def render_svg_image(self, size: Size) -> bytes | None:
        """Render an SVG to PNG bytes."""
        try:
            source_path = self.path_config.resolve_project_path(size.source)
            png_data = cairosvg.svg2png(
                url=str(source_path),
                output_width=size.width,
                output_height=size.height,
            )
            return png_data
        except Exception as e:
            logging.getLogger(__name__).error(f"Error rendering SVG image {size.source}: {e}")
            return None

    def render_raster_image(self, size: Size) -> Image.Image | None:
        """Render a raster image (PNG, JPG, etc.) with resizing."""
        try:
            source_path = self.path_config.resolve_project_path(size.source)
            img = Image.open(source_path)

            # Calculate scaling to fit within target size while maintaining aspect ratio
            width_ratio = size.width / img.width
            height_ratio = size.height / img.height
            scale_factor = min(width_ratio, height_ratio)
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create output image with proper mode
            mode = "RGBA" if img.mode == "RGBA" else "RGB"
            bg_color = (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)
            output_img = Image.new(mode, (size.width, size.height), bg_color)

            # Center the resized image
            x = (size.width - new_width) // 2
            y = (size.height - new_height) // 2
            output_img.paste(resized_img, (x, y))

            return output_img
        except Exception as e:
            logging.getLogger(__name__).error(f"Error rendering raster image {size.source}: {e}")
            return None

    def _load_font_with_fallback(self, font_path: Path, target_size: int) -> ImageFont.FreeTypeFont | None:
        """Load font with fallback sizes if target size fails."""
        for font_size in [target_size, int(target_size * 0.9), int(target_size * 0.8)]:
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except (OSError, IOError):
                continue
        return None


# ============================================================================
# Task Processing - Worker process logic
# ============================================================================


@dataclass
class GenerationTask:
    """Represents a single image generation task."""

    task_id: int
    theme_name: str
    image_name: str
    size_name: str
    size: Size


@dataclass
class GenerationResult:
    """Result of an image generation task."""

    task_id: int
    success: bool
    theme_name: str
    yaml_entry: dict[str, str] | None = None
    error_message: str | None = None


class WorkerContext:
    """Context shared across worker process, initialized once per worker."""

    def __init__(self, path_config: PathConfig, config: Config, log_queue, log_level):
        # Setup logging for this worker
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
        root_logger.setLevel(log_level)

        self.path_config = path_config
        self.config = config
        self.renderer = ImageRenderer(path_config)


# Module-level worker context (set during initialization)
_worker_context: WorkerContext | None = None


def init_worker_process(path_config: PathConfig, config: Config, log_queue, log_level):
    """Initialize worker process with necessary context."""
    global _worker_context
    _worker_context = WorkerContext(path_config, config, log_queue, log_level)


def process_generation_task(task: GenerationTask) -> GenerationResult:
    """Process a single image generation task in a worker process."""
    global _worker_context
    if _worker_context is None:
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            theme_name=task.theme_name,
            error_message="Worker not initialized"
        )

    logger = logging.getLogger(__name__)
    logger.debug(
        f"[{task.task_id}] Processing {task.theme_name}.images.{task.image_name}.sizes.{task.size_name}"
    )

    try:
        # Determine output path
        output_path = _worker_context.path_config.get_image_output_path(
            task.theme_name, task.image_name, task.size.slug
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Render the appropriate image type
        image_data = None
        if task.size.text:  # Font-based image
            logger.debug(
                f"[{task.task_id}]   Type: Text, Size: {task.size.width}x{task.size.height}, Color: {task.size.color}"
            )
            font = _worker_context.config.themes[task.theme_name].fonts[task.size.text.font]
            img = _worker_context.renderer.render_font_image(task.size, font)
            if img:
                img.save(output_path, "PNG", optimize=True)
                image_data = True

        elif task.size.source:  # File-based image
            source_path = _worker_context.path_config.resolve_project_path(task.size.source)
            if source_path.suffix.lower() == ".svg":
                logger.debug(
                    f"[{task.task_id}]   Type: SVG, Size: {task.size.width}x{task.size.height}, Source: '{task.size.source}'"
                )
                png_bytes = _worker_context.renderer.render_svg_image(task.size)
                if png_bytes:
                    output_path.write_bytes(png_bytes)
                    image_data = True
            else:
                logger.debug(
                    f"[{task.task_id}]   Type: Raster, Size: {task.size.width}x{task.size.height}, Source: '{task.size.source}'"
                )
                img = _worker_context.renderer.render_raster_image(task.size)
                if img:
                    img.save(output_path, "PNG", optimize=True)
                    image_data = True

        if image_data:
            logger.debug(f"[{task.task_id}] Generated: {output_path}")
            yaml_entry = {
                "file": str(_worker_context.path_config.yaml_base / task.theme_name / task.size.slug / f"{task.image_name}.png"),
                "id": f"image_{task.image_name}_{task.size.slug}",
                "type": "RGB565",
                "transparency": "alpha_channel",
            }
            return GenerationResult(
                task_id=task.task_id,
                success=True,
                theme_name=task.theme_name,
                yaml_entry=yaml_entry
            )
        else:
            logger.warning(
                f"[{task.task_id}] Failed to generate {task.theme_name}.images.{task.image_name}.sizes.{task.size_name}"
            )
            return GenerationResult(
                task_id=task.task_id,
                success=False,
                theme_name=task.theme_name,
                error_message="Image rendering failed"
            )

    except Exception as e:
        logger.error(f"[{task.task_id}] Exception during generation: {e}", exc_info=True)
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            theme_name=task.theme_name,
            error_message=str(e)
        )


# ============================================================================
# Task Orchestration - Main application logic
# ============================================================================


class TaskBuilder:
    """Builds the list of generation tasks from configuration."""

    def __init__(self, config: Config):
        self.config = config

    def build_tasks(self) -> list[GenerationTask]:
        """Build all image generation tasks from the configuration."""
        tasks = []
        task_id = 0

        for theme_name, theme in self.config.themes.items():
            if not theme.generate:
                continue

            for image_name, image in theme.images.items():
                for size_name, size in image.sizes.items():
                    if size.skip:
                        continue

                    tasks.append(GenerationTask(
                        task_id=task_id,
                        theme_name=theme_name,
                        image_name=image_name,
                        size_name=size_name,
                        size=size
                    ))
                    task_id += 1

        return tasks


class YamlConfigWriter:
    """Writes YAML configuration files for generated images."""

    def __init__(self, path_config: PathConfig):
        self.path_config = path_config

    def write_theme_configs(self, results: list[GenerationResult]):
        """Write YAML configuration files for all themes."""
        logger = logging.getLogger(__name__)

        # Group results by theme
        theme_data: dict[str, list[dict]] = {}
        for result in results:
            if result.success and result.yaml_entry:
                if result.theme_name not in theme_data:
                    theme_data[result.theme_name] = []
                theme_data[result.theme_name].append(result.yaml_entry)

        # Write a YAML file for each theme
        for theme_name, entries in theme_data.items():
            logger.info(f"Generating YAML configuration for theme '{theme_name}'")
            output_path = self.path_config.get_yaml_output_path(theme_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write("# Auto-generated file. Re-run image generator to regenerate.\n")
                yaml.dump({"image": entries}, f, default_flow_style=False)


class GenerationOrchestrator:
    """Orchestrates the entire image generation process."""

    def __init__(
        self,
        path_config: PathConfig,
        config: Config,
        log_queue,
        log_level,
        num_workers: int
    ):
        self.path_config = path_config
        self.config = config
        self.log_queue = log_queue
        self.log_level = log_level
        self.num_workers = num_workers
        self.task_builder = TaskBuilder(config)
        self.yaml_writer = YamlConfigWriter(path_config)

    def run(self) -> int:
        """Execute the image generation process."""
        logger = logging.getLogger(__name__)

        logger.info("Image Generator for ESPHome")
        logger.info(f"Using {self.num_workers} processes for generation.")
        logger.info("=" * 50)

        # Build all tasks
        tasks = self.task_builder.build_tasks()
        logger.info(f"Total images to generate: {len(tasks)}")

        if not tasks:
            logger.warning("No tasks to process")
            return 0

        # Execute tasks in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=init_worker_process,
            initargs=(self.path_config, self.config, self.log_queue, self.log_level),
        ) as executor:
            results = list(executor.map(process_generation_task, tasks))

        # Analyze results
        successful = sum(1 for r in results if r.success)
        logger.info(f"Generation complete: {successful}/{len(tasks)} images generated.")

        # Write YAML configuration files
        self.yaml_writer.write_theme_configs(results)

        return 0 if successful == len(tasks) else 1


# ============================================================================
# Application Entry Point
# ============================================================================


class ColorFormatter(logging.Formatter):
    """Custom formatter with color output for console logging."""

    WHITE = "\x1b[37m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, datefmt="%H:%M:%S"):
        super().__init__(datefmt=datefmt)

    def format(self, record):
        asctime = self.formatTime(record, self.datefmt)
        tenths = int(record.msecs / 100)
        timestamp = f"{self.WHITE}{asctime}.{tenths}{self.RESET}"

        level_color = self.LEVEL_COLORS.get(record.levelno, self.WHITE)
        levelname = f"{level_color}{record.levelname:<8}{self.RESET}"

        message = record.getMessage()
        message_color = f"{level_color}{message}{self.RESET}"

        # Handle multi-line messages with proper indentation
        lines = message_color.splitlines()
        if len(lines) > 1:
            indentation = " " * (len(asctime) + 2 + 8 + 4)
            indented_lines = [lines[0]] + [f"{indentation}{line}" for line in lines[1:]]
            message_color = "\n".join(indented_lines)

        return f"{timestamp} - {levelname} - {message_color}"


class Application:
    """Main application class."""

    def __init__(self, args):
        self.args = args
        self.path_config = PathConfig(Path(__file__))

        # Setup logging infrastructure
        self.manager = multiprocessing.Manager()
        self.log_queue = self.manager.Queue(-1)
        self.log_level = logging.DEBUG if args.verbose else logging.INFO

        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for main process and listener."""
        # Main process logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(logging.handlers.QueueHandler(self.log_queue))
        root_logger.setLevel(self.log_level)

        # Listener for all worker logs
        formatter = ColorFormatter(datefmt="%H:%M:%S")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.listener = logging.handlers.QueueListener(self.log_queue, stream_handler)

    def run(self) -> int:
        """Run the application."""
        logger = logging.getLogger(__name__)
        self.listener.start()

        try:
            # Load configuration
            config = self._load_config()
            if config is None:
                return 1

            # Run the generation orchestrator
            orchestrator = GenerationOrchestrator(
                path_config=self.path_config,
                config=config,
                log_queue=self.log_queue,
                log_level=self.log_level,
                num_workers=self.args.threads
            )
            return orchestrator.run()

        except Exception as e:
            logger.exception(f"An unhandled exception occurred: {e}")
            return 1
        finally:
            self.listener.stop()

    def _load_config(self) -> Config | None:
        """Load and parse the configuration file."""
        logger = logging.getLogger(__name__)

        try:
            with open(self.path_config.image_config_path, "r") as config_file:
                raw_config = yaml.safe_load(config_file)
                return Config(raw_config)

        except FileNotFoundError:
            logger.error(f"Error: '{self.path_config.image_config_path}' not found.")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in '{self.path_config.image_config_path}': {e}")
            return None
        except ValueError as e:
            logger.error(f"Error reading configuration in '{self.path_config.image_config_path}': {e}")
            return None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Image and ESPHome image configuration generator"
    )
    _ = parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    default_threads = max(2, (os.cpu_count() or 4) - 2)
    _ = parser.add_argument(
        "--threads",
        type=int,
        default=default_threads,
        help=f"Number of parallel image generation tasks (default: {default_threads})",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    app = Application(args)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
