# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Alex Popescu (@pspsalex)

from dataclasses import dataclass

@dataclass(frozen=True)
class IconSize:
    """Icon size configuration."""
    name: str
    pixels: int
    description: str = ""


@dataclass(frozen=True)
class Icon:
    """Icon definition with metadata."""
    name: str
    unicode_char: str
    description: str = ""
    category: str = "general"


class IconDefinitions:
    """Centralized icon definitions."""

    # Icon size definitions
    SIZES = [
        IconSize("small", 32, "Status bar and small UI elements"),
        IconSize("medium", 48, "Standard UI icons"),
        IconSize("large", 128, "Large display icons"),
        IconSize("status", 32, "Status indicators"),
    ]

    # Icon definitions organized by category
    ICONS = [
        # Navigation
        Icon("forward", "\U0000e5e1", "Forward navigation", "navigation"),
        Icon("backward", "\U0000e2ea", "Back navigation", "navigation"),
        Icon("home", "\U0000e88a", "Home button", "navigation"),

        # WiFi states
        Icon("wifi_unknown", "\U0000f0ef", "WiFi unknown state", "connectivity"),
        Icon("wifi_0", "\U0000f063", "WiFi no signal", "connectivity"),
        Icon("wifi_1", "\U0000f0b0", "WiFi 1 bar", "connectivity"),
        Icon("wifi_2", "\U0000ebe4", "WiFi 2 bars", "connectivity"),
        Icon("wifi_3", "\U0000ebd6", "WiFi 3 bars", "connectivity"),
        Icon("wifi_4", "\U0000ebe1", "WiFi 4 bars", "connectivity"),
        Icon("wifi_5", "\U0000e1d8", "WiFi full signal", "connectivity"),

        # Network
        Icon("lan", "\U0000eb2f", "LAN connection", "connectivity"),
        Icon("cloud_off", "\U0000e2c1", "Cloud disconnected", "connectivity"),
        Icon("cloud_done", "\U0000e2bf", "Cloud connected", "connectivity"),

        # Battery states
        Icon("battery_full", "\U0000e1a4", "Battery full", "power"),
        Icon("battery_6_bar", "\U0000ebd2", "Battery 6/6", "power"),
        Icon("battery_5_bar", "\U0000ebd4", "Battery 5/6", "power"),
        Icon("battery_4_bar", "\U0000ebe2", "Battery 4/6", "power"),
        Icon("battery_3_bar", "\U0000ebdd", "Battery 3/6", "power"),
        Icon("battery_2_bar", "\U0000ebe0", "Battery 2/6", "power"),
        Icon("battery_1_bar", "\U0000ebd9", "Battery 1/6", "power"),
        Icon("battery_0_bar", "\U0000ebdc", "Battery empty", "power"),
        Icon("battery_unknown", "\U0000e1a6", "Battery unknown", "power"),

        # Weather
        Icon("sunny", "\U0000e81a", "Sunny weather", "weather"),
        Icon("partly_cloudy_day", "\U0000f172", "Partly cloudy", "weather"),
        Icon("cloud", "\U0000e2bd", "Cloudy weather", "weather"),
        Icon("thunderstorm", "\U0000ebdb", "Storm weather", "weather"),
        Icon("night", "\U0000ef44", "Night/moon", "weather"),
        Icon("twilight", "\U0000e1c6", "Twilight", "weather"),
        Icon("wind", "\U0000efd8", "Windy weather", "weather"),
        Icon("scene", "\U0000e2a7", "Scene/landscape", "weather"),

        # Controls
        Icon("vert_align_top", "\U0000e316", "Up arrow", "controls"),
        Icon("vert_align_bottom", "\U0000e313", "Down arrow", "controls"),
        Icon("lightbulb_on", "\U0000e0f0", "Light on", "controls"),
        Icon("lightbulb_off", "\U0000e9b8", "Light off", "controls"),
        Icon("stop", "\U0000e047", "Stop button", "controls"),
        Icon("hourglass", "\U0000ef64", "Loading/waiting", "controls"),
    ]
