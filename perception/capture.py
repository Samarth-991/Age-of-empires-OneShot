"""Screenshot capture with perceptual-hash deduplication.

Uses PowerShell's System.Drawing.Graphics.CopyFromScreen via subprocess —
the only approach that works reliably from WSL2 where the Linux display
stack cannot access the Windows screen.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import imagehash
from PIL import Image
from config import SCREENSHOT_OUT_DIR, SCREENSHOT_OUT_DIR_WINDOWS
# Windows path where PowerShell writes screenshots
_WIN_SCREENSHOTS = SCREENSHOT_OUT_DIR_WINDOWS
# Corresponding WSL path (shared between Windows and WSL2)
_WSL_SCREENSHOTS = Path(SCREENSHOT_OUT_DIR)


def capture_screenshot(bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """Capture the Windows primary screen via PowerShell and return a PIL Image (RGB).

    Uses System.Drawing.Graphics.CopyFromScreen — WSL2 compatible because
    powershell.exe runs natively on Windows and writes to a path both sides share.
    The bbox parameter is accepted for API compatibility but ignored; PowerShell
    always captures the full primary screen.

    Raises:
        subprocess.CalledProcessError: if PowerShell exits with a non-zero code.
        FileNotFoundError: if the saved PNG cannot be found at the expected WSL path.
    """
    _WSL_SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    filename = f"cap_{int(time.time() * 1000)}.png"
    windows_path = f"{_WIN_SCREENSHOTS}\\{filename}"
    wsl_path = _WSL_SCREENSHOTS / filename

    powershell_cmd = (
        "[Reflection.Assembly]::LoadWithPartialName('System.Drawing') | Out-Null; "
        "[Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
        "$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; "
        "$graphics = [System.Drawing.Graphics]::FromImage($bmp); "
        "$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); "
        f"$bmp.Save('{windows_path}', [System.Drawing.Imaging.ImageFormat]::Png); "
        "$graphics.Dispose(); $bmp.Dispose();"
    )
    subprocess.run(
        ["powershell.exe", "-Command", powershell_cmd],
        check=True,
        capture_output=True,
    )
    return Image.open(wsl_path).convert("RGB")


def get_aoe2_window_bbox() -> Optional[Tuple[int, int, int, int]]:
    """Best-effort: locate the AoE2 window and return (left, top, width, height).

    Returns None when the window is not found or pygetwindow is not installed.
    When the game runs fullscreen the PowerShell capture covers the entire
    primary screen, so a bbox is not strictly required.
    """
    try:
        import pygetwindow as gw
    except Exception:
        return None

    for title in ("Age of Empires", "AoE2"):
        matches = [w for w in gw.getWindowsWithTitle(title) if w.width > 0]
        if matches:
            w = matches[0]
            return (w.left, w.top, w.width, w.height)
    return None


def save_screenshot(img: Image.Image, out_dir: str = SCREENSHOT_OUT_DIR) -> str:
    """Save image to out_dir with millisecond timestamp; return the file path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"screenshot_{int(time.time() * 1000)}.png")
    img.save(path)
    return path


class CaptureLoop:
    """Manages one-shot screenshot capture with perceptual-hash deduplication.

    Attributes:
        out_dir: Directory where screenshots are written.
        interval: Seconds between captures (informational; not enforced here).
        hash_threshold: Perceptual-hash diff below which VLM call is skipped.
    """

    def __init__(
        self,
        out_dir: str = SCREENSHOT_OUT_DIR,
        interval: float = 3.0,
        hash_threshold: int = 8,
    ) -> None:
        self.out_dir = out_dir
        self.interval = interval
        self.hash_threshold = hash_threshold
        self._last_hash: Optional[imagehash.ImageHash] = None
        self._last_mini_hash: Optional[imagehash.ImageHash] = None

    def _hash(self, img: Image.Image, size: Tuple[int, int] = (64, 64)) -> imagehash.ImageHash:
        return imagehash.average_hash(img.resize(size).convert("L"))

    def _crop_minimap(self, img: Image.Image) -> Image.Image:
        """Heuristically crop the minimap region (bottom-right ~18% of screen)."""
        w, h = img.size
        mw, mh = int(w * 0.18), int(h * 0.18)
        return img.crop((w - mw, h - mh, w, h))

    def run_once(self) -> Tuple[str, Image.Image, bool]:
        """Capture one frame and evaluate whether the VLM should be called.

        Returns:
            (path, img, needs_vlm) — needs_vlm is False when the frame is
            visually too similar to the previous one to warrant a VLM call.
        """
        img = capture_screenshot()
        path = save_screenshot(img, out_dir=self.out_dir)

        current_hash = self._hash(img)
        mini_hash = self._hash(self._crop_minimap(img))

        needs_vlm = True
        if self._last_hash is not None:
            diff = current_hash - self._last_hash
            mini_diff = (
                mini_hash - self._last_mini_hash
                if self._last_mini_hash is not None
                else 99
            )
            if diff <= self.hash_threshold and mini_diff <= (self.hash_threshold // 2):
                needs_vlm = False

        self._last_hash = current_hash
        self._last_mini_hash = mini_hash
        return path, img, needs_vlm
