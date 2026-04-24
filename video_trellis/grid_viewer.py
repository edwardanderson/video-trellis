"""
Real-time grid viewer for video shots using Pygame.

This module provides dynamic, GPU-accelerated viewing of video shots
arranged in a grid without requiring re-encoding.
"""

import bisect
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable, cast
from collections import deque

import pygame
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image

RESAMPLE_BILINEAR = Image.Resampling.BILINEAR

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    """Parse a SubRip (.srt) file into a sorted list of (start_sec, end_sec, text) tuples.

    HTML tags such as <i> and <b> are stripped.  Entries are returned sorted by
    start time so binary search can be used at render time.
    """
    def _srt_ts_to_sec(ts: str) -> float:
        # Format: HH:MM:SS,mmm
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s

    entries: List[Tuple[float, float, str]] = []
    text = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        # Find the timecode line (contains " --> ")
        tc_idx = next((i for i, l in enumerate(lines) if " --> " in l), None)
        if tc_idx is None:
            continue
        tc_line = lines[tc_idx]
        arrow_pos = tc_line.index(" --> ")
        try:
            start = _srt_ts_to_sec(tc_line[:arrow_pos].strip())
            end = _srt_ts_to_sec(tc_line[arrow_pos + 5:].strip().split()[0])
        except (ValueError, IndexError):
            continue
        raw_text = "\n".join(lines[tc_idx + 1:]).strip()
        clean_text = _HTML_TAG_RE.sub("", raw_text).strip()
        if clean_text:
            entries.append((start, end, clean_text))

    entries.sort(key=lambda e: e[0])
    return entries


def timecode_to_seconds(timecode: str) -> float:
    """Convert HH:MM:SS.mmm timecode to seconds."""
    if isinstance(timecode, (int, float)):
        return float(timecode)
    
    parts = timecode.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(timecode)


class ShotListManifest:
    """Represents a shot list manifest (JSON or CSV format)."""
    
    def __init__(self, manifest_path: Path):
        """
        Load a shot list manifest (supports both JSON and CSV formats).
        
        Parameters
        ----------
        manifest_path : Path
            Path to the manifest file (JSON or CSV)
        """
        self.manifest_path = manifest_path
        self.data = {}
        self.shots = []
        self._load_manifest()
    
    def _load_manifest(self):
        """Load and parse the manifest file."""
        if self.manifest_path.suffix.lower() == '.csv':
            self._load_csv()
        else:
            self._load_json()
    
    def _load_csv(self):
        """Load CSV format (scenedetect output)."""
        try:
            with open(self.manifest_path, 'r') as f:
                # Skip the first line (timecode list summary) if it exists
                lines = f.readlines()
                
                # Find the header row (starts with "Scene Number")
                header_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('Scene Number'):
                        header_idx = i
                        break
                
                # Parse CSV starting from the header row
                reader = csv.DictReader(lines[header_idx:])
                for row in reader:
                    if row.get('Start Time (seconds)') and row.get('End Time (seconds)'):
                        try:
                            start = float(row['Start Time (seconds)'])
                            end = float(row['End Time (seconds)'])
                            self.shots.append((start, end))
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            raise ValueError(f"Failed to parse CSV manifest: {e}")
    
    def _load_json(self):
        """Load and parse the manifest JSON file."""
        try:
            with open(self.manifest_path, 'r') as f:
                self.data = json.load(f)
            
            # Parse shots from manifest
            # pyscenedetect format typically has 'shot_list' or similar structure
            if 'shot_list' in self.data:
                self.shots = self.data['shot_list']
            elif 'scenes' in self.data:
                self.shots = self.data['scenes']
            elif isinstance(self.data, list):
                self.shots = self.data
            else:
                # Try to extract any list from the data
                for key, value in self.data.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], (dict, list, tuple)):
                            self.shots = value
                            break
                            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse manifest JSON: {e}")
    
    def get_shot_timecodes(self) -> List[Tuple[float, float]]:
        """
        Extract start and end times for each shot.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (start_time, end_time) tuples in seconds
        """
        timecodes = []
        
        for shot in self.shots:
            if isinstance(shot, dict):
                # Try common key patterns
                if 'start' in shot and 'end' in shot:
                    start = timecode_to_seconds(shot['start'])
                    end = timecode_to_seconds(shot['end'])
                elif 'start_time' in shot and 'end_time' in shot:
                    start = timecode_to_seconds(shot['start_time'])
                    end = timecode_to_seconds(shot['end_time'])
                elif 'timecode' in shot:
                    # Single timecode, assume frame or seconds
                    tc = timecode_to_seconds(shot['timecode'])
                    start = tc
                    end = tc + 0.1  # Arbitrary 100ms duration for display
                else:
                    continue
            elif isinstance(shot, (list, tuple)) and len(shot) >= 2:
                start = timecode_to_seconds(shot[0])
                end = timecode_to_seconds(shot[1])
            else:
                continue
                
            timecodes.append((start, end))
        
        return timecodes


def calculate_grid_layout(
    shot_count: int,
    canvas_width: int,
    canvas_height: int,
    video_width: int,
    video_height: int,
    allow_padding: bool = True,
) -> Tuple[Tuple[int, int], Tuple[int, int], int, int]:
    """
    Calculate optimal grid layout for shots.
    
    Parameters
    ----------
    shot_count : int
        Number of shots to display
    canvas_width : int
        Canvas/window width in pixels
    canvas_height : int
        Canvas/window height in pixels
    video_width : int
        Original video width
    video_height : int
        Original video height
    allow_padding : bool
        Whether to preserve aspect ratio with padding
    
    Returns
    -------
    Tuple[Tuple[int, int], Tuple[int, int], int, int]
        ((cell_width, cell_height), (rows, cols), pad_x, pad_y)
    """
    video_aspect = video_width / video_height
    
    best_scale = 0.0
    best_dims = (0, 0)
    best_layout = (0, 0)
    
    for rows in range(1, shot_count + 1):
        cols = math.ceil(shot_count / rows)
        
        cell_w = canvas_width / cols
        cell_h = canvas_height / rows
        
        if allow_padding:
            # Preserve aspect ratio inside cell
            scale = min(cell_w / video_width, cell_h / video_height)
            scaled_w = video_width * scale
            scaled_h = video_height * scale
        else:
            # Require exact aspect match
            cell_aspect = cell_w / cell_h
            if not math.isclose(cell_aspect, video_aspect, rel_tol=1e-6):
                continue
            
            scale = cell_w / video_width
            scaled_w = cell_w
            scaled_h = cell_h
        
        if scale > best_scale:
            best_scale = scale
            best_dims = (int(scaled_w), int(scaled_h))
            best_layout = (rows, cols)
    
    # Calculate padding
    rows, cols = best_layout
    target_w, target_h = best_dims
    grid_w = cols * target_w
    grid_h = rows * target_h
    
    if allow_padding:
        pad_x = (canvas_width - grid_w) // 2
        pad_y = (canvas_height - grid_h) // 2
    else:
        pad_x = 0
        pad_y = 0
    
    return best_dims, best_layout, pad_x, pad_y


class ContextMenu:
    """A right-click context menu drawn with Pygame primitives."""

    ITEM_HEIGHT = 30
    PADDING_X = 14
    PADDING_Y = 6
    BG_COLOR = (28, 28, 28)
    BORDER_COLOR = (85, 85, 85)
    HOVER_COLOR = (55, 100, 200)
    TEXT_COLOR = (220, 220, 220)
    _FONT_SIZE = 22

    def __init__(self, x: int, y: int, items: List[str], payload: Dict[str, object]):
        self.x = x
        self.y = y
        self.items = items
        self.payload = payload
        self.hovered_idx: int = -1
        self._font: Optional[pygame.font.Font] = None

    def _get_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.Font(None, self._FONT_SIZE)
        return self._font

    def _menu_size(self) -> Tuple[int, int]:
        font = self._get_font()
        w = max((font.size(item)[0] for item in self.items), default=80) + self.PADDING_X * 2
        h = len(self.items) * self.ITEM_HEIGHT + self.PADDING_Y * 2
        return w, h

    def menu_rect(self) -> pygame.Rect:
        w, h = self._menu_size()
        return pygame.Rect(self.x, self.y, w, h)

    def clamp_to_screen(self, screen_w: int, screen_h: int) -> None:
        w, h = self._menu_size()
        if self.x + w > screen_w:
            self.x = screen_w - w
        if self.y + h > screen_h:
            self.y = screen_h - h

    def update_hover(self, mouse_pos: Tuple[int, int]) -> None:
        mx, my = mouse_pos
        r = self.menu_rect()
        if not r.collidepoint(mx, my):
            self.hovered_idx = -1
            return
        rel_y = my - self.y - self.PADDING_Y
        idx = rel_y // self.ITEM_HEIGHT
        self.hovered_idx = idx if 0 <= idx < len(self.items) else -1

    def item_at(self, mouse_pos: Tuple[int, int]) -> Optional[int]:
        """Return the index of the item under mouse_pos, or None."""
        self.update_hover(mouse_pos)
        return self.hovered_idx if self.hovered_idx >= 0 else None

    def draw(self, surface: pygame.Surface) -> None:
        font = self._get_font()
        r = self.menu_rect()
        pygame.draw.rect(surface, self.BG_COLOR, r, border_radius=4)
        pygame.draw.rect(surface, self.BORDER_COLOR, r, width=1, border_radius=4)
        for i, item in enumerate(self.items):
            item_rect = pygame.Rect(
                self.x,
                self.y + self.PADDING_Y + i * self.ITEM_HEIGHT,
                r.width,
                self.ITEM_HEIGHT,
            )
            if i == self.hovered_idx:
                pygame.draw.rect(surface, self.HOVER_COLOR, item_rect, border_radius=3)
            text_surf = font.render(item, True, self.TEXT_COLOR)
            text_y = item_rect.y + (self.ITEM_HEIGHT - text_surf.get_height()) // 2
            surface.blit(text_surf, (self.x + self.PADDING_X, text_y))


class ShotViewer:
    """
    Full-screen overlay that plays a single shot with audio.

    Frame decoding is done on-demand at the native video resolution (scaled to
    fit the overlay panel) and cached by 50 ms time bucket.  Audio is extracted
    to a temp WAV file on a background thread; pygame.mixer picks it up once
    the file is ready.
    """

    OVERLAY_ALPHA = 180
    PANEL_MARGIN_FRAC = 0.07
    CTRL_BAR_H = 44
    CLOSE_BTN_SIZE = 30

    def __init__(
        self,
        shot_idx: int,
        start_time: float,
        end_time: float,
        video_path: Path,
        fps: float,
        screen_w: int,
        screen_h: int,
        total_shots: int = 1,
    ):
        self.shot_idx = shot_idx
        self.total_shots = total_shots
        self.start_time = start_time
        self.end_time = end_time
        self.shot_duration = max(end_time - start_time, 0.001)
        self.fps = fps
        self.playback_time: float = 0.0
        self._quantum = 0.05  # 50 ms buckets
        self._frame_cache: Dict[int, np.ndarray] = {}

        # Panel geometry
        mx = int(screen_w * self.PANEL_MARGIN_FRAC)
        my = int(screen_h * self.PANEL_MARGIN_FRAC)
        self.panel_rect = pygame.Rect(mx, my, screen_w - 2 * mx, screen_h - 2 * my)

        # Load video clip (audio stripped — we handle audio separately)
        self._clip = VideoFileClip(str(video_path), audio=False)
        vid_w, vid_h = self._clip.w, self._clip.h
        avail_w = self.panel_rect.width
        avail_h = self.panel_rect.height - self.CTRL_BAR_H - 8
        scale = min(avail_w / vid_w, avail_h / vid_h)
        self.frame_w = max(1, int(vid_w * scale))
        self.frame_h = max(1, int(vid_h * scale))

        # Pre-build fonts
        self._font = pygame.font.Font(None, 22)

        # Wall-clock timer drives playback_time for drift-free A/V sync
        self._wall_start: float = time.perf_counter()

        # Audio state — extraction runs on a background thread
        self._audio_channel: Optional[pygame.mixer.Channel] = None
        self._audio_sound: Optional[pygame.mixer.Sound] = None
        self._audio_tmpfile: Optional[str] = None
        self._audio_ready = False  # set True by background thread when file is written
        self._audio_thread = threading.Thread(
            target=self._extract_audio,
            args=(video_path,),
            daemon=True,
        )
        self._audio_thread.start()

    # ------------------------------------------------------------------
    # Audio

    def _extract_audio(self, video_path: Path) -> None:
        """Background thread: extract shot audio segment to a temp WAV."""
        try:
            audio_clip = VideoFileClip(str(video_path))
            shot_clip = audio_clip.subclipped(self.start_time, self.end_time)
            if shot_clip.audio is None:
                audio_clip.close()
                return
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            shot_clip.audio.write_audiofile(tmp.name, logger=None)
            audio_clip.close()
            self._audio_tmpfile = tmp.name
            self._audio_ready = True
        except Exception as e:
            print(f"[ShotViewer] Audio extraction failed: {e}", flush=True)

    def _tick_audio(self) -> None:
        """Called each frame from the main thread: start audio once ready."""
        if not self._audio_ready or self._audio_channel is not None:
            return
        try:
            self._audio_sound = pygame.mixer.Sound(self._audio_tmpfile)
            # Reset the wall clock so video snaps to t=0, matching audio
            # which always starts from the beginning of the WAV file.
            self._wall_start = time.perf_counter()
            self._audio_channel = self._audio_sound.play(loops=-1)
            self._audio_ready = False
        except Exception as e:
            print(f"[ShotViewer] Audio play failed: {e}", flush=True)
            self._audio_ready = False

    # ------------------------------------------------------------------
    # Playback

    def update(self) -> None:
        # Viewer playback is self-timed (independent of grid playback clock).
        self.playback_time = (time.perf_counter() - self._wall_start) % self.shot_duration
        self._tick_audio()

    def _get_current_frame(self) -> Optional[np.ndarray]:
        bucket = int(self.playback_time / self._quantum)
        if bucket in self._frame_cache:
            return self._frame_cache[bucket]
        try:
            t = self.start_time + min(self.playback_time, self.shot_duration - 0.001)
            raw = self._clip.get_frame(t)  # H×W×3 uint8
            if raw is None:
                return None
            img = Image.fromarray(cast(np.ndarray, raw)).resize((self.frame_w, self.frame_h), RESAMPLE_BILINEAR)
            frame = np.array(img)
            self._frame_cache[bucket] = frame
            # Evict oldest entries to cap memory use
            if len(self._frame_cache) > 400:
                del self._frame_cache[min(self._frame_cache)]
            return frame
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Rendering

    @property
    def close_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.right - self.CLOSE_BTN_SIZE - 8,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE,
            self.CLOSE_BTN_SIZE,
        )

    @property
    def prev_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.left + 8,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE + 16,
            self.CLOSE_BTN_SIZE,
        )

    @property
    def next_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.left + 8 + self.CLOSE_BTN_SIZE + 16 + 6,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE + 16,
            self.CLOSE_BTN_SIZE,
        )

    def draw(self, surface: pygame.Surface) -> None:
        sw, sh = surface.get_size()

        # Semi-transparent dim layer
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, self.OVERLAY_ALPHA))
        surface.blit(dim, (0, 0))

        # Panel background
        pygame.draw.rect(surface, (18, 18, 18), self.panel_rect, border_radius=6)
        pygame.draw.rect(surface, (75, 75, 75), self.panel_rect, width=2, border_radius=6)

        # Video frame
        frame = self._get_current_frame()
        if frame is not None:
            frame_surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            fx = self.panel_rect.x + (self.panel_rect.width - self.frame_w) // 2
            fy = self.panel_rect.y + 6
            surface.blit(frame_surf, (fx, fy))

        # Progress bar
        bar_margin = 16
        bar_x = self.panel_rect.x + bar_margin
        bar_w = self.panel_rect.width - 2 * bar_margin
        bar_h = 5
        bar_y = self.panel_rect.bottom - self.CTRL_BAR_H + 18
        pygame.draw.rect(surface, (55, 55, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * (self.playback_time / self.shot_duration))
        if fill_w > 0:
            pygame.draw.rect(surface, (90, 155, 255), (bar_x, bar_y, fill_w, bar_h), border_radius=3)

        # Info label
        lbl = self._font.render(
            f"Shot {self.shot_idx + 1}  {self.playback_time:.1f}s / {self.shot_duration:.1f}s"
            f"  [ESC to close]",
            True,
            (190, 190, 190),
        )
        surface.blit(lbl, (bar_x, self.panel_rect.bottom - self.CTRL_BAR_H + 2))

        # Close button
        cbr = self.close_btn_rect
        pygame.draw.rect(surface, (110, 35, 35), cbr, border_radius=4)
        x_lbl = self._font.render("x", True, (255, 255, 255))
        surface.blit(
            x_lbl,
            (cbr.x + (cbr.width - x_lbl.get_width()) // 2,
             cbr.y + (cbr.height - x_lbl.get_height()) // 2),
        )

        # Prev button
        has_prev = self.shot_idx > 0
        pbr = self.prev_btn_rect
        pygame.draw.rect(surface, (45, 45, 75) if has_prev else (30, 30, 40), pbr, border_radius=4)
        p_lbl = self._font.render("◀ Prev", True, (210, 210, 255) if has_prev else (80, 80, 90))
        surface.blit(
            p_lbl,
            (pbr.x + (pbr.width - p_lbl.get_width()) // 2,
             pbr.y + (pbr.height - p_lbl.get_height()) // 2),
        )

        # Next button
        has_next = self.shot_idx < self.total_shots - 1
        nbr = self.next_btn_rect
        pygame.draw.rect(surface, (45, 45, 75) if has_next else (30, 30, 40), nbr, border_radius=4)
        n_lbl = self._font.render("Next ▶", True, (210, 210, 255) if has_next else (80, 80, 90))
        surface.blit(
            n_lbl,
            (nbr.x + (nbr.width - n_lbl.get_width()) // 2,
             nbr.y + (nbr.height - n_lbl.get_height()) // 2),
        )

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop audio, release the clip, delete temp file."""
        if self._audio_channel is not None:
            try:
                self._audio_channel.stop()
            except Exception:
                pass
        if self._audio_sound is not None:
            try:
                self._audio_sound.stop()
            except Exception:
                pass
        if self._audio_tmpfile and os.path.exists(self._audio_tmpfile):
            try:
                os.unlink(self._audio_tmpfile)
            except Exception:
                pass
        try:
            self._clip.close()
        except Exception:
            pass


class SequenceViewer:
    """Full-screen overlay that plays an ordered sequence of selected shots."""

    OVERLAY_ALPHA = ShotViewer.OVERLAY_ALPHA
    PANEL_MARGIN_FRAC = ShotViewer.PANEL_MARGIN_FRAC
    CTRL_BAR_H = ShotViewer.CTRL_BAR_H
    CLOSE_BTN_SIZE = ShotViewer.CLOSE_BTN_SIZE

    def __init__(
        self,
        shot_indices: List[int],
        shot_timecodes: List[Tuple[float, float]],
        video_path: Path,
        fps: float,
        screen_w: int,
        screen_h: int,
        start_position: int = 0,
    ):
        if not shot_indices:
            raise ValueError("SequenceViewer requires at least one selected shot")

        self.shot_indices = shot_indices[:]
        self.shot_timecodes = shot_timecodes
        self.fps = fps
        self.current_sequence_idx = max(0, min(start_position, len(self.shot_indices) - 1))
        self.sequence_time = 0.0
        self.current_local_time = 0.0
        self.current_shot_idx = self.shot_indices[self.current_sequence_idx]
        self._quantum = 0.05
        self._frame_cache: Dict[Tuple[int, int], np.ndarray] = {}

        self.segment_durations = [
            max(self.shot_timecodes[shot_idx][1] - self.shot_timecodes[shot_idx][0], 0.001)
            for shot_idx in self.shot_indices
        ]
        self.segment_offsets: List[float] = []
        running_total = 0.0
        for duration in self.segment_durations:
            self.segment_offsets.append(running_total)
            running_total += duration
        self.sequence_duration = max(running_total, 0.001)

        mx = int(screen_w * self.PANEL_MARGIN_FRAC)
        my = int(screen_h * self.PANEL_MARGIN_FRAC)
        self.panel_rect = pygame.Rect(mx, my, screen_w - 2 * mx, screen_h - 2 * my)

        self._clip = VideoFileClip(str(video_path), audio=False)
        vid_w, vid_h = self._clip.w, self._clip.h
        avail_w = self.panel_rect.width
        avail_h = self.panel_rect.height - self.CTRL_BAR_H - 8
        scale = min(avail_w / vid_w, avail_h / vid_h)
        self.frame_w = max(1, int(vid_w * scale))
        self.frame_h = max(1, int(vid_h * scale))
        self._font = pygame.font.Font(None, 22)

        # Sequence audio: extract each selected segment to a temp WAV in background.
        self._audio_channel: Optional[pygame.mixer.Channel] = None
        self._audio_sound: Optional[pygame.mixer.Sound] = None
        self._audio_tmpfiles: Dict[int, str] = {}
        self._audio_extract_stop = False
        self._audio_playing_sequence_idx: Optional[int] = None
        self._audio_lock = threading.Lock()
        self._audio_thread = threading.Thread(
            target=self._extract_sequence_audio,
            args=(video_path,),
            daemon=True,
        )
        self._audio_thread.start()

        self._wall_start = time.perf_counter() - self.segment_offsets[self.current_sequence_idx]
        self._resolve_sequence_position(self.segment_offsets[self.current_sequence_idx])

    def _extract_sequence_audio(self, video_path: Path) -> None:
        """Background thread: write one WAV temp file per selected shot segment."""
        try:
            audio_source = VideoFileClip(str(video_path))
            if audio_source.audio is None:
                audio_source.close()
                return

            for seq_idx, shot_idx in enumerate(self.shot_indices):
                if self._audio_extract_stop:
                    break
                start_time, end_time = self.shot_timecodes[shot_idx]
                if end_time <= start_time:
                    continue

                seg = audio_source.subclipped(start_time, end_time)
                if seg.audio is None:
                    continue

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                seg.audio.write_audiofile(tmp.name, logger=None)
                with self._audio_lock:
                    self._audio_tmpfiles[seq_idx] = tmp.name

            audio_source.close()
        except Exception as e:
            print(f"[SequenceViewer] Audio extraction failed: {e}", flush=True)

    def _tick_audio(self) -> None:
        """Ensure audio for current sequence segment is playing."""
        target_seq_idx = self.current_sequence_idx
        with self._audio_lock:
            audio_path = self._audio_tmpfiles.get(target_seq_idx)
        if not audio_path:
            return

        if (
            self._audio_playing_sequence_idx == target_seq_idx
            and self._audio_channel is not None
            and self._audio_channel.get_busy()
        ):
            return

        try:
            if self._audio_channel is not None:
                self._audio_channel.stop()
            self._audio_sound = pygame.mixer.Sound(audio_path)
            self._audio_channel = self._audio_sound.play(loops=0)
            self._audio_playing_sequence_idx = target_seq_idx
        except Exception as e:
            print(f"[SequenceViewer] Audio play failed: {e}", flush=True)

    @property
    def close_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.right - self.CLOSE_BTN_SIZE - 8,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE,
            self.CLOSE_BTN_SIZE,
        )

    @property
    def prev_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.left + 8,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE + 16,
            self.CLOSE_BTN_SIZE,
        )

    @property
    def next_btn_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.panel_rect.left + 8 + self.CLOSE_BTN_SIZE + 16 + 6,
            self.panel_rect.top + 8,
            self.CLOSE_BTN_SIZE + 16,
            self.CLOSE_BTN_SIZE,
        )

    def can_go_prev(self) -> bool:
        return self.current_sequence_idx > 0

    def can_go_next(self) -> bool:
        return self.current_sequence_idx < len(self.shot_indices) - 1

    def step(self, delta: int) -> None:
        next_idx = self.current_sequence_idx + delta
        next_idx = max(0, min(next_idx, len(self.shot_indices) - 1))
        self._wall_start = time.perf_counter() - self.segment_offsets[next_idx]
        self._resolve_sequence_position(self.segment_offsets[next_idx])
        self._audio_playing_sequence_idx = None
        self._tick_audio()

    def _resolve_sequence_position(self, sequence_time: float) -> None:
        self.sequence_time = sequence_time
        for seq_idx, offset in enumerate(self.segment_offsets):
            duration = self.segment_durations[seq_idx]
            if sequence_time < offset + duration or seq_idx == len(self.segment_offsets) - 1:
                self.current_sequence_idx = seq_idx
                self.current_shot_idx = self.shot_indices[seq_idx]
                self.current_local_time = max(0.0, min(sequence_time - offset, duration - 0.001))
                return

    def update(self) -> None:
        sequence_time = (time.perf_counter() - self._wall_start) % self.sequence_duration
        self._resolve_sequence_position(sequence_time)
        self._tick_audio()

    def _get_current_frame(self) -> Optional[np.ndarray]:
        bucket = int(self.current_local_time / self._quantum)
        cache_key = (self.current_sequence_idx, bucket)
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key]

        try:
            start_time, end_time = self.shot_timecodes[self.current_shot_idx]
            shot_duration = max(end_time - start_time, 0.001)
            t = start_time + min(self.current_local_time, shot_duration - 0.001)
            raw = self._clip.get_frame(t)
            if raw is None:
                return None
            img = Image.fromarray(cast(np.ndarray, raw)).resize((self.frame_w, self.frame_h), RESAMPLE_BILINEAR)
            frame = np.array(img)
            self._frame_cache[cache_key] = frame
            if len(self._frame_cache) > 800:
                oldest_key = next(iter(self._frame_cache))
                del self._frame_cache[oldest_key]
            return frame
        except Exception:
            return None

    def draw(self, surface: pygame.Surface) -> None:
        sw, sh = surface.get_size()
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, self.OVERLAY_ALPHA))
        surface.blit(dim, (0, 0))

        pygame.draw.rect(surface, (18, 18, 18), self.panel_rect, border_radius=6)
        pygame.draw.rect(surface, (75, 75, 75), self.panel_rect, width=2, border_radius=6)

        frame = self._get_current_frame()
        if frame is not None:
            frame_surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            fx = self.panel_rect.x + (self.panel_rect.width - self.frame_w) // 2
            fy = self.panel_rect.y + 6
            surface.blit(frame_surf, (fx, fy))

        bar_margin = 16
        bar_x = self.panel_rect.x + bar_margin
        bar_w = self.panel_rect.width - 2 * bar_margin
        bar_h = 5
        bar_y = self.panel_rect.bottom - self.CTRL_BAR_H + 18
        pygame.draw.rect(surface, (55, 55, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * (self.sequence_time / self.sequence_duration))
        if fill_w > 0:
            pygame.draw.rect(surface, (90, 155, 255), (bar_x, bar_y, fill_w, bar_h), border_radius=3)

        shot_duration = self.segment_durations[self.current_sequence_idx]
        lbl = self._font.render(
            f"Sequence {self.current_sequence_idx + 1}/{len(self.shot_indices)} | Shot {self.current_shot_idx + 1}"
            f" | {self.current_local_time:.1f}s / {shot_duration:.1f}s | [ESC to close]",
            True,
            (190, 190, 190),
        )
        surface.blit(lbl, (bar_x, self.panel_rect.bottom - self.CTRL_BAR_H + 2))

        cbr = self.close_btn_rect
        pygame.draw.rect(surface, (110, 35, 35), cbr, border_radius=4)
        x_lbl = self._font.render("x", True, (255, 255, 255))
        surface.blit(
            x_lbl,
            (cbr.x + (cbr.width - x_lbl.get_width()) // 2,
             cbr.y + (cbr.height - x_lbl.get_height()) // 2),
        )

        has_prev = self.can_go_prev()
        pbr = self.prev_btn_rect
        pygame.draw.rect(surface, (45, 45, 75) if has_prev else (30, 30, 40), pbr, border_radius=4)
        p_lbl = self._font.render("◀ Prev", True, (210, 210, 255) if has_prev else (80, 80, 90))
        surface.blit(
            p_lbl,
            (pbr.x + (pbr.width - p_lbl.get_width()) // 2,
             pbr.y + (pbr.height - p_lbl.get_height()) // 2),
        )

        has_next = self.can_go_next()
        nbr = self.next_btn_rect
        pygame.draw.rect(surface, (45, 45, 75) if has_next else (30, 30, 40), nbr, border_radius=4)
        n_lbl = self._font.render("Next ▶", True, (210, 210, 255) if has_next else (80, 80, 90))
        surface.blit(
            n_lbl,
            (nbr.x + (nbr.width - n_lbl.get_width()) // 2,
             nbr.y + (nbr.height - n_lbl.get_height()) // 2),
        )

    def close(self) -> None:
        self._audio_extract_stop = True
        if self._audio_channel is not None:
            try:
                self._audio_channel.stop()
            except Exception:
                pass
        if self._audio_sound is not None:
            try:
                self._audio_sound.stop()
            except Exception:
                pass
        with self._audio_lock:
            audio_paths = list(self._audio_tmpfiles.values())
            self._audio_tmpfiles.clear()
        for path in audio_paths:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
        try:
            self._clip.close()
        except Exception:
            pass


class GridViewer:
    """Real-time grid viewer for video shots using Pygame."""
    
    def __init__(
        self,
        video_path: Path,
        shot_timecodes: List[Tuple[float, float]],
        window_width: int = 1920,
        window_height: int = 1080,
        fps: Optional[int] = None,
        allow_padding: bool = True,
        fullscreen: bool = False,
        maximized: bool = False,
        verbose: bool = False,
        subtitle_path: Optional[Path] = None,
    ):
        """
        Initialize the grid viewer.
        
        Parameters
        ----------
        video_path : Path
            Path to the input video file
        shot_timecodes : List[Tuple[float, float]]
            List of (start_time, end_time) tuples for each shot
        window_width : int
            Window width in pixels
        window_height : int
            Window height in pixels
        fps : Optional[int]
            Frames per second. If None, uses video's FPS
        allow_padding : bool
            Whether to preserve aspect ratio with padding
        fullscreen : bool
            Open viewer in fullscreen mode
        """
        self.video_path = video_path
        self.shot_timecodes = shot_timecodes
        self.window_width = window_width
        self.window_height = window_height
        self.allow_padding = allow_padding
        self.fullscreen = fullscreen
        self.maximized = maximized
        self.verbose = verbose

        # Subtitle support
        if subtitle_path is not None:
            self.subtitles: List[Tuple[float, float, str]] = parse_srt(subtitle_path)
        else:
            self.subtitles = []
        self._subtitle_starts: List[float] = [s for s, _e, _t in self.subtitles]
        # Zoom level at which subtitles become visible (tiles must be large enough to read text).
        self.subtitle_zoom_threshold: float = 1.5
        # Font cache keyed by pixel size, populated lazily during rendering.
        self._subtitle_font_cache: Dict[int, pygame.font.Font] = {}
        # Caption layout cache: (text, draw_w, draw_h, font_size) -> pre-rendered overlay surface.
        self._subtitle_overlay_cache: Dict[Tuple[str, int, int, int], pygame.Surface] = {}
        self._subtitle_overlay_cache_max_entries: int = 512
        self.subtitles_visible: bool = True
        self.video = VideoFileClip(str(video_path), audio=False)
        self.video_width = int(self.video.w)
        self.video_height = int(self.video.h)
        self.fps = fps or self.video.fps
        
        # Pygame setup
        pygame.init()
        self._display_flags = 0
        if self.fullscreen:
            self._display_flags = pygame.FULLSCREEN
            self.display = pygame.display.set_mode((0, 0), self._display_flags)
            self.window_width, self.window_height = self.display.get_size()
        else:
            self._display_flags = pygame.RESIZABLE
            self.display = pygame.display.set_mode((window_width, window_height), self._display_flags)
            if self.maximized:
                # Best-effort maximize on pygame2/SDL2 backends.
                try:
                    from pygame._sdl2 import Window
                    Window.from_display_module().maximize()
                    self.window_width, self.window_height = self.display.get_size()
                except Exception:
                    pass
        pygame.display.set_caption(f"Video Trellis Viewer - {video_path.name}")
        self.clock = pygame.time.Clock()

        # Calculate grid layout
        self.cell_dims, self.layout, self.pad_x, self.pad_y = calculate_grid_layout(
            len(shot_timecodes),
            self.window_width,
            self.window_height,
            self.video_width,
            self.video_height,
            allow_padding=allow_padding,
        )
        
        # Playback state
        self.current_time = 0.0
        self.playing = True
        self.max_duration = max(end for _, end in shot_timecodes) if shot_timecodes else 0
        # Quantize to 100ms buckets to maximize cache reuse at playback FPS.
        self.cache_time_quantum = 0.1
        self.shot_bucket_counts: Dict[int, int] = {
            idx: max(1, int((end - start) /  self.cache_time_quantum) + 1)
            for idx, (start, end) in enumerate(shot_timecodes)
            if end - start > 0
        }

        # Cache memory model: one RGB frame per bucket, per shot cell resolution.
        cell_w, cell_h = self.cell_dims
        self.bytes_per_cached_frame = cell_w * cell_h * 3
        self.total_bucket_frames = sum(self.shot_bucket_counts.values())
        self.estimated_full_cache_bytes = self.total_bucket_frames * self.bytes_per_cached_frame
        self.available_ram_bytes = self._get_available_ram_bytes()
        if self.available_ram_bytes is not None:
            # Performance-first default: keep full cache resident when it fits with basic reserve headroom.
            reserve_floor = 1_500_000_000  # Keep ~1.5GB free for OS/background work.
            ram_budget = max(int(self.available_ram_bytes * 0.85), self.available_ram_bytes - reserve_floor)
            self.full_cache_in_ram_mode = self.estimated_full_cache_bytes <= max(ram_budget, 0)
        else:
            # Fallback when RAM probing fails: allow resident mode for moderate cache footprints.
            self.full_cache_in_ram_mode = self.estimated_full_cache_bytes <= 2_000_000_000

        preload_pref = os.environ.get("VIDEO_TRELLIS_FULL_CACHE_PRELOAD", "auto").strip().lower()
        if preload_pref in {"1", "true", "yes", "on"}:
            self.full_cache_preload_mode = True
        elif preload_pref in {"0", "false", "no", "off"}:
            self.full_cache_preload_mode = False
        else:
            # Auto default is performance-first: preload whenever resident cache mode is enabled.
            self.full_cache_preload_mode = self.full_cache_in_ram_mode

        # Ready queue per shot (frame bucket -> decoded frame) consumed by render loop.
        self.frame_buffer: Dict[int, Dict[int, np.ndarray]] = {}
        self.frame_read_order: Dict[int, deque[int]] = {}
        self.frame_buffer_max_per_shot = 72
        # Extra cache for zoomed-in rendering (shot -> {(frame_idx, size_key): frame}).
        self.hires_frame_buffer: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
        self.hires_frame_read_order: Dict[int, deque[Tuple[int, int]]] = {}
        hires_target_quantum = min(self.cache_time_quantum, 1.0 / max(float(self.fps), 1.0))
        self.hires_frame_buffer_max_per_shot = max(120, int(math.ceil(6.0 / hires_target_quantum)))
        # Hi-res decode is background-only to keep the UI thread responsive.
        self.hires_decode_enabled_zoom_base = 1.25
        self.hires_decode_enabled_zoom_min = 1.05
        self.hires_decode_enabled_zoom_max = 1.35
        self.hires_decode_reference_cell_pixels = 160_000
        self.hires_decode_overscan = 1.35
        self.hires_decode_max_pixels = 3_145_728  # ~2048x1536 max per visible tile
        self.hires_fullres_viewport_threshold = 0.80
        self.hires_cache_prune_zoom_ratio = 2.0
        self.hires_prefetch_budget_per_cycle = 18
        self.hires_prefetch_budget_max_boosted = 40
        self.hires_cursor_hint_lead_frames = 5
        self.hires_cursor_hint_weight_boost = 0.55
        # Keep the base cache coarse for RAM efficiency, but let zoomed playback track source FPS.
        self.hires_time_quantum = hires_target_quantum
        self.hires_prefetch_max_lead_frames = 16
        self.hires_nearest_fallback_radius = 6
        self.hires_prefetch_cursor = 0
        self._visible_hires_requests: List[Tuple[int, int, Tuple[int, int]]] = []
        self._visible_shot_indices: List[int] = []
        self._visible_shot_weights: Dict[int, int] = {}
        self.warmup_frames_per_shot = 20  # Pre-decode first 2.0s per shot with 100ms buckets
        self._cache_lock = threading.Lock()
        self.prefetch_base_depth = 6
        self.prefetch_max_depth = 12
        self.prefetch_decode_ewma_ms = 0.0
        self.latest_ready_frame: Dict[int, np.ndarray] = {}
        self.latest_ready_idx: Dict[int, int] = {}
        self.prefetch_cursor = 0
        self.prefetch_cycle_decode_budget = 8
        self.prefetch_lag_cap_buckets = 8
        self.prefetch_emergency_lag_buckets = 12
        if self.full_cache_in_ram_mode:
            # Spend more decode budget per cycle when we are building a resident cache.
            self.prefetch_cycle_decode_budget = 14
        
        # Performance tracking
        self.frame_times = deque(maxlen=300)  # Track last 300 frame times for FPS calculation
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_fallback_reuses = 0
        self.decode_misses = 0
        self.queue_starvations = 0
        self.decode_times = deque(maxlen=100)  # Track decode times
        self.render_times = deque(maxlen=100)  # Track render times
        self.perf_log_interval = 10  # Log stats every N frames
        self.last_perf_log_time = time.perf_counter()
        self.shot_qos_log_every_n_frames = 50
        self.shot_qos_fresh_window_seconds = 5
        self.shot_qos_last_signature = None
        self.shot_stats = {
            "exact_hits": [0] * len(self.shot_timecodes),
            "fallback_hits": [0] * len(self.shot_timecodes),
            "starvations": [0] * len(self.shot_timecodes),
            "lag_sum": [0] * len(self.shot_timecodes),
            "lag_max": [0] * len(self.shot_timecodes),
            "lag_samples": [0] * len(self.shot_timecodes),
            "decode_ms_sum": [0.0] * len(self.shot_timecodes),
            "decode_samples": [0] * len(self.shot_timecodes),
            "exact_hit_frames": [deque() for _ in range(len(self.shot_timecodes))],
        }
        
        # Background pre-decoding thread for smooth playback
        self.prefetch_stop_flag = False
        self.prefetch_thread = None
        self.prefetch_video = None

        # Interactive overlay state
        self._context_menu: Optional[ContextMenu] = None
        self._shot_viewer: Optional[ShotViewer] = None
        self._sequence_viewer: Optional[SequenceViewer] = None
        self.selected_shot_indices: List[int] = []
        self._export_thread: Optional[threading.Thread] = None
        self._export_status_lock = threading.Lock()
        self._export_status: str = ""
        self._resize_pending_size: Optional[Tuple[int, int]] = None
        self._resize_last_event_ts: float = 0.0
        self._resize_settle_seconds: float = 0.25
        self._last_left_click_ts: float = 0.0
        self._last_left_click_shot_idx: Optional[int] = None
        self._double_click_seconds: float = 0.30

        # Grid zoom state (applies only to the background shot grid).
        self.grid_zoom = 1.0
        self.grid_zoom_min = 1.0
        # Allow zooming until a single cell fills the screen edge-to-edge on at least one axis.
        cell_w, cell_h = self.cell_dims
        self.grid_zoom_max = max(
            self.window_width / max(1, cell_w),
            self.window_height / max(1, cell_h),
        )
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0
        self.pan_pixels_per_second = 900.0
        self.edge_pan_margin = 28
        # Allow a small overscroll when zoomed-in so edge tile content can be
        # brought fully into view instead of hard-clamping exactly at grid bounds.
        self.pan_overscroll_fraction = 0.08
        self.pan_overscroll_min_px = 40
        self.pan_overscroll_max_px = 220

        # Warmup progress (updated by background thread, read by render loop)
        self._warmup_progress: int = 0   # frames decoded so far
        self._warmup_total: int = 0      # total frames to decode
        self._warmup_done: bool = False  # set True when warmup thread finishes

        if self.verbose:
            if self.full_cache_in_ram_mode:
                available = self._format_bytes(self.available_ram_bytes or 0)
                estimated = self._format_bytes(self.estimated_full_cache_bytes)
                print(f"RAM cache mode: ON (estimated {estimated} for full bucket cache, available {available})")
                if self.full_cache_preload_mode:
                    print("RAM preload mode: ON (pre-decoding all quantized buckets before playback)")
                else:
                    print("RAM preload mode: OFF (set VIDEO_TRELLIS_FULL_CACHE_PRELOAD=1 to force)")
            else:
                estimated = self._format_bytes(self.estimated_full_cache_bytes)
                print(f"RAM cache mode: OFF (estimated full bucket cache {estimated}; using bounded per-shot cache)")

    def _get_available_ram_bytes(self) -> Optional[int]:
        """Best-effort available RAM detection without external dependencies."""
        # Linux fast-path.
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024
        except Exception:
            pass

        # POSIX fallback.
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size) * int(avail_pages)
        except Exception:
            return None

    def _format_bytes(self, value: int) -> str:
        """Format bytes into compact human-readable units."""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(max(value, 0))
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)}{unit}"
                return f"{size:.1f}{unit}"
            size /= 1024.0

        return f"{value}B"

    def _buffer_limit_for_shot(self, shot_idx: int) -> int:
        """Return max cached buckets for a shot under current memory policy."""
        if self.full_cache_in_ram_mode:
            return self.shot_bucket_counts.get(shot_idx, self.frame_buffer_max_per_shot)
        return self.frame_buffer_max_per_shot

    def _frame_idx_to_time(self, shot_idx: int, frame_idx: int) -> float:
        """Convert a quantized frame bucket index to absolute source video time."""
        return self._frame_idx_to_time_for_quantum(shot_idx, frame_idx, self.cache_time_quantum)

    def _frame_idx_to_time_for_quantum(self, shot_idx: int, frame_idx: int, quantum: float) -> float:
        """Convert a quantized frame bucket index to absolute source video time for a given quantum."""
        start_time, end_time = self.shot_timecodes[shot_idx]
        shot_duration = end_time - start_time
        if shot_duration <= 0:
            return start_time
        time_offset = min(frame_idx * quantum, max(shot_duration - 0.001, 0.0))
        return start_time + time_offset

    def _frame_idx_for_quantum(self, shot_idx: int, time_offset: float, quantum: float) -> int:
        """Return frame bucket index for a shot at a custom quantum."""
        start_time, end_time = self.shot_timecodes[shot_idx]
        shot_duration = end_time - start_time
        if shot_duration <= 0:
            return 0
        bucket_count = max(1, int(shot_duration / quantum) + 1)
        return int(time_offset / quantum) % bucket_count

    def _get_playback_frame_idx(self, shot_idx: int) -> int:
        """Get currently requested frame bucket for a shot based on playback clock."""
        start_time, end_time = self.shot_timecodes[shot_idx]
        shot_duration = end_time - start_time
        if shot_duration <= 0:
            return 0
        time_in_shot = self.current_time % shot_duration
        bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
        return int(time_in_shot / self.cache_time_quantum) % bucket_count

    def _enqueue_ready_frame(self, shot_idx: int, frame_idx: int, frame: np.ndarray):
        """Insert decoded frame into per-shot ready queue with bounded size."""
        with self._cache_lock:
            if shot_idx not in self.frame_buffer:
                self.frame_buffer[shot_idx] = {}
                self.frame_read_order[shot_idx] = deque()

            if frame_idx in self.frame_buffer[shot_idx]:
                self.frame_buffer[shot_idx][frame_idx] = frame
                return

            self.frame_buffer[shot_idx][frame_idx] = frame
            self.frame_read_order[shot_idx].append(frame_idx)
            self.latest_ready_frame[shot_idx] = frame
            self.latest_ready_idx[shot_idx] = frame_idx

            buffer_limit = self._buffer_limit_for_shot(shot_idx)
            while len(self.frame_buffer[shot_idx]) > buffer_limit:
                oldest_idx = self.frame_read_order[shot_idx].popleft()
                self.frame_buffer[shot_idx].pop(oldest_idx, None)

    def _decode_ready_frame(self, shot_idx: int, frame_idx: int, source_video=None) -> Optional[np.ndarray]:
        """Decode one quantized frame bucket and return resized frame."""
        frame_time = self._frame_idx_to_time(shot_idx, frame_idx)
        decode_start = time.perf_counter()
        frame = self._get_or_decode_frame(
            shot_idx,
            frame_idx,
            frame_time,
            source_video=source_video,
            target_dims=self.cell_dims,
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000
        self.decode_times.append(decode_ms)
        self.shot_stats["decode_ms_sum"][shot_idx] += decode_ms
        self.shot_stats["decode_samples"][shot_idx] += 1
        if self.prefetch_decode_ewma_ms == 0.0:
            self.prefetch_decode_ewma_ms = decode_ms
        else:
            self.prefetch_decode_ewma_ms = self.prefetch_decode_ewma_ms * 0.9 + decode_ms * 0.1
        return frame

    def _record_shot_lag(self, shot_idx: int, requested_idx: int, ready_idx: Optional[int]):
        """Track per-shot freshness lag in quantized bucket units."""
        if ready_idx is None:
            return
        bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
        lag = (requested_idx - ready_idx) % bucket_count
        self.shot_stats["lag_sum"][shot_idx] += lag
        self.shot_stats["lag_samples"][shot_idx] += 1
        if lag > self.shot_stats["lag_max"][shot_idx]:
            self.shot_stats["lag_max"][shot_idx] = lag

    def _prefetch_depth_target(self) -> int:
        """Adapt target queue depth from observed decode cost."""
        if self.prefetch_decode_ewma_ms <= 0:
            return self.prefetch_base_depth
        extra = int(min(max(self.prefetch_decode_ewma_ms / 8.0, 0), self.prefetch_max_depth - self.prefetch_base_depth))
        return min(self.prefetch_max_depth, self.prefetch_base_depth + extra)

    def _bucket_lag(self, playback_idx: int, ready_idx: int, bucket_count: int) -> int:
        """Return forward distance from latest-ready to playback index in circular bucket space."""
        if bucket_count <= 0:
            return 0
        return (playback_idx - ready_idx) % bucket_count
    
    def _warmup_frame_caches(self):
        """Pre-decode multiple frames per shot to populate initial cache.

        Runs on a background thread.  Progress is reported via
        self._warmup_progress / self._warmup_total so the render loop can
        display a live progress bar.  Sets self._warmup_done when finished.
        """
        total_to_decode = 0
        for start_time, end_time in self.shot_timecodes:
            shot_duration = end_time - start_time
            if shot_duration <= 0:
                continue
            max_bucket_idx = int(shot_duration / self.cache_time_quantum)
            if self.full_cache_preload_mode:
                total_to_decode += max_bucket_idx + 1
            else:
                total_to_decode += min(self.warmup_frames_per_shot, max_bucket_idx + 1)
        self._warmup_total = total_to_decode
        decoded_count = 0

        for shot_idx, (start_time, end_time) in enumerate(self.shot_timecodes):
            shot_duration = end_time - start_time
            if shot_duration <= 0:
                continue

            max_bucket_idx = int(shot_duration / self.cache_time_quantum)
            if self.full_cache_preload_mode:
                frames_per_shot = max_bucket_idx + 1
            else:
                frames_per_shot = min(self.warmup_frames_per_shot, max_bucket_idx + 1)

            # Pre-decode multiple frames from the beginning of each shot
            self.frame_buffer[shot_idx] = {}
            self.frame_read_order[shot_idx] = deque()

            for frame_idx in range(frames_per_shot):
                frame = self._decode_ready_frame(shot_idx, frame_idx)
                if frame is not None:
                    self._enqueue_ready_frame(shot_idx, frame_idx, frame)

                decoded_count += 1
                self._warmup_progress = decoded_count

            # Progress indicator
            if self.verbose and (shot_idx + 1) % max(1, len(self.shot_timecodes) // 10) == 0:
                print(f"  Warmed {shot_idx + 1}/{len(self.shot_timecodes)} shots (~{decoded_count}/{total_to_decode} frames)")

        self._warmup_done = True
        if self.verbose:
            print("✓ Frame caches ready")
    
    def _start_prefetch_thread(self):
        """Start background thread for continuous frame pre-decoding."""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            if self.prefetch_video is None:
                # Use a dedicated decoder so background work never blocks foreground playback.
                self.prefetch_video = VideoFileClip(str(self.video_path), audio=False)
            self.prefetch_stop_flag = False
            self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.prefetch_thread.start()
    
    def _stop_prefetch_thread(self):
        """Stop the background prefetch thread."""
        self.prefetch_stop_flag = True
        if self.prefetch_thread is not None:
            self.prefetch_thread.join(timeout=1.0)
        self.prefetch_thread = None
        if self.prefetch_video is not None:
            self.prefetch_video.close()
            self.prefetch_video = None
    
    def _prefetch_loop(self):
        """Background thread that keeps per-shot ready queues filled ahead of playback."""
        while not self.prefetch_stop_flag:
            try:
                target_depth = self._prefetch_depth_target()
                total_shot_count = len(self.shot_timecodes)
                if total_shot_count == 0:
                    time.sleep(0.01)
                    continue

                with self._cache_lock:
                    visible_shots = list(self._visible_shot_indices)
                    visible_weights = dict(self._visible_shot_weights)

                hires_zoom_threshold = self._current_hires_decode_enabled_zoom()
                if self.grid_zoom >= hires_zoom_threshold and visible_shots:
                    # At high zoom, focus base prefetch on what is actually visible.
                    visible_unique = self._ordered_visible_shots(visible_shots, visible_weights)
                    local_count = len(visible_unique)
                    order = visible_unique
                else:
                    local_count = total_shot_count
                    order = [
                        (self.prefetch_cursor + i) % total_shot_count
                        for i in range(total_shot_count)
                    ]

                decoded_this_cycle = 0
                # Keep budget bounded to avoid one long cycle starving tail shots.
                cycle_budget = max(4, min(self.prefetch_cycle_decode_budget, local_count))
                critical_depth = min(4, target_depth)
                decoded_shots = set()

                # Build a snapshot so we can prioritize stale shots with large lag first.
                shot_states = []
                with self._cache_lock:
                    for shot_idx in order:
                        start_time, end_time = self.shot_timecodes[shot_idx]
                        shot_duration = end_time - start_time
                        if shot_duration <= 0:
                            continue

                        bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
                        playback_idx = self._get_playback_frame_idx(shot_idx)
                        needed_now = [
                            (playback_idx + offset) % bucket_count
                            for offset in range(critical_depth)
                        ]

                        ready = self.frame_buffer.get(shot_idx, {})
                        missing_now = [idx for idx in needed_now if idx not in ready]

                        latest_idx = self.latest_ready_idx.get(shot_idx)
                        if latest_idx is None:
                            lag_buckets = bucket_count
                        else:
                            lag_buckets = self._bucket_lag(playback_idx, latest_idx, bucket_count)

                        needed = [
                            (playback_idx + offset) % bucket_count
                            for offset in range(target_depth)
                        ]
                        missing = [idx for idx in needed if idx not in ready]

                        shot_states.append({
                            "shot_idx": shot_idx,
                            "lag_buckets": lag_buckets,
                            "missing_now": missing_now,
                            "missing": missing,
                        })

                def decode_target(shot_idx: int, frame_idx: int) -> bool:
                    nonlocal decoded_this_cycle
                    if self.prefetch_stop_flag or decoded_this_cycle >= cycle_budget:
                        return False
                    frame = self._decode_ready_frame(shot_idx, frame_idx, source_video=self.prefetch_video)
                    if frame is None:
                        return False
                    self._enqueue_ready_frame(shot_idx, frame_idx, frame)
                    decoded_this_cycle += 1
                    decoded_shots.add(shot_idx)
                    return True

                # Pass 0: cap staleness first so no shot drifts too far behind.
                emergency = [
                    s for s in shot_states
                    if s["missing_now"] and s["lag_buckets"] >= self.prefetch_emergency_lag_buckets
                ]
                emergency.sort(key=lambda s: s["lag_buckets"], reverse=True)
                for state in emergency:
                    if not decode_target(state["shot_idx"], state["missing_now"][0]):
                        break

                # Pass 1: strict fairness. Give at most one near-playback decode per shot in round-robin order.
                for shot_idx in order:
                    if self.prefetch_stop_flag or decoded_this_cycle >= cycle_budget:
                        break
                    if shot_idx in decoded_shots:
                        continue

                    start_time, end_time = self.shot_timecodes[shot_idx]
                    shot_duration = end_time - start_time
                    if shot_duration <= 0:
                        continue

                    bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
                    playback_idx = self._get_playback_frame_idx(shot_idx)
                    needed_now = [
                        (playback_idx + offset) % bucket_count
                        for offset in range(critical_depth)
                    ]

                    with self._cache_lock:
                        ready = self.frame_buffer.get(shot_idx, {})
                        missing_now = [idx for idx in needed_now if idx not in ready]

                    if not missing_now:
                        continue

                    decode_target(shot_idx, missing_now[0])

                # Pass 2: prioritize shots still above lag cap.
                if decoded_this_cycle < cycle_budget:
                    stale = [
                        s for s in shot_states
                        if s["shot_idx"] not in decoded_shots and s["missing_now"] and s["lag_buckets"] >= self.prefetch_lag_cap_buckets
                    ]
                    stale.sort(key=lambda s: s["lag_buckets"], reverse=True)
                    for state in stale:
                        if not decode_target(state["shot_idx"], state["missing_now"][0]):
                            break

                # Pass 3: use remaining budget for deeper lookahead, still round-robin fair.
                if decoded_this_cycle < cycle_budget:
                    for shot_idx in order:
                        if self.prefetch_stop_flag or decoded_this_cycle >= cycle_budget:
                            break
                        if shot_idx in decoded_shots:
                            continue

                        start_time, end_time = self.shot_timecodes[shot_idx]
                        shot_duration = end_time - start_time
                        if shot_duration <= 0:
                            continue

                        bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
                        playback_idx = self._get_playback_frame_idx(shot_idx)
                        needed = [
                            (playback_idx + offset) % bucket_count
                            for offset in range(target_depth)
                        ]

                        with self._cache_lock:
                            ready = self.frame_buffer.get(shot_idx, {})
                            missing = [idx for idx in needed if idx not in ready]

                        if not missing:
                            continue

                        decode_target(shot_idx, missing[0])

                # Advance cursor by budget to cover the full shot set uniformly.
                if local_count > 0:
                    self.prefetch_cursor = (self.prefetch_cursor + max(decoded_this_cycle, 1)) % local_count

                hires_budget = self.hires_prefetch_budget_per_cycle
                decoded_hires = 0
                if hires_budget > 0:
                    with self._cache_lock:
                        hires_requests = list(self._visible_hires_requests)
                        visible_weights = dict(self._visible_shot_weights)
                    if hires_requests:
                        ordered = sorted(
                            hires_requests,
                            key=lambda request: visible_weights.get(request[0], 0),
                            reverse=True,
                        )
                        # Multiple visible tiles can emit duplicate requests for the same key.
                        seen_reqs = set()
                        deduped = []
                        for req in ordered:
                            if req in seen_reqs:
                                continue
                            seen_reqs.add(req)
                            deduped.append(req)
                        ordered = deduped
                        if self.grid_zoom >= hires_zoom_threshold:
                            hires_budget = self._hires_budget_for_visibility(len(ordered), visible_weights)
                            visible_count = max(1, len(visible_weights))
                            if visible_count <= 2:
                                hires_budget = min(len(ordered), max(hires_budget, 24))
                            elif visible_count <= 4:
                                hires_budget = min(len(ordered), max(hires_budget, 16))
                        else:
                            # Cursor hint mode: keep pre-decode cheap before full zoom.
                            hires_budget = min(2, len(ordered))
                    else:
                        ordered = []

                    for shot_idx, frame_idx, target_dims in ordered:
                        if self.prefetch_stop_flag or hires_budget <= 0:
                            break
                        if target_dims[0] <= self.cell_dims[0] and target_dims[1] <= self.cell_dims[1]:
                            continue
                        if self._get_cached_hires_frame(shot_idx, frame_idx, target_dims) is not None:
                            continue
                        frame_time = self._frame_idx_to_time_for_quantum(shot_idx, frame_idx, self.hires_time_quantum)
                        frame = self._decode_hires_frame(
                            shot_idx,
                            frame_idx,
                            frame_time,
                            target_dims,
                            source_video=self.prefetch_video,
                        )
                        if frame is not None:
                            hires_budget -= 1
                            decoded_hires += 1

                    if hires_requests:
                        self.hires_prefetch_cursor = (self.hires_prefetch_cursor + max(decoded_hires, 1)) % len(hires_requests)
                
                # Adaptive sleep: when no work was done, back off to reduce idle CPU burn.
                total_decoded = decoded_this_cycle + decoded_hires
                if total_decoded == 0:
                    time.sleep(0.015)
                elif total_decoded <= 2:
                    time.sleep(0.008)
                else:
                    time.sleep(0.004)
            except Exception:
                # Silently ignore errors in background thread
                pass
    
    def _size_key_for_dims(self, dims: Tuple[int, int]) -> int:
        """Quantize target dimensions to a compact key for cache reuse."""
        w, h = dims
        # 16px buckets are enough to avoid excessive cache fragmentation.
        qw = max(1, int(round(w / 16.0)) * 16)
        qh = max(1, int(round(h / 16.0)) * 16)
        return (qw << 16) | qh

    def _dims_for_size_key(self, size_key: int) -> Tuple[int, int]:
        """Decode packed size key back to quantized width/height."""
        qw = (size_key >> 16) & 0xFFFF
        qh = size_key & 0xFFFF
        return max(1, qw), max(1, qh)

    def _decode_hires_frame(
        self,
        shot_idx: int,
        frame_idx: int,
        frame_time: float,
        target_dims: Tuple[int, int],
        source_video=None,
    ) -> Optional[np.ndarray]:
        """Decode one frame for zoomed-in view and cache it by size tier."""
        size_key = self._size_key_for_dims(target_dims)
        with self._cache_lock:
            by_shot = self.hires_frame_buffer.setdefault(shot_idx, {})
            key = (frame_idx, size_key)
            if key in by_shot:
                return by_shot[key]

        frame = self._get_or_decode_frame(
            shot_idx,
            frame_idx,
            frame_time,
            source_video=source_video,
            target_dims=target_dims,
        )
        if frame is None:
            return None

        with self._cache_lock:
            by_shot = self.hires_frame_buffer.setdefault(shot_idx, {})
            order = self.hires_frame_read_order.setdefault(shot_idx, deque())
            key = (frame_idx, size_key)
            if key not in by_shot:
                by_shot[key] = frame
                order.append(key)
            else:
                by_shot[key] = frame

            while len(by_shot) > self.hires_frame_buffer_max_per_shot:
                oldest = order.popleft()
                by_shot.pop(oldest, None)

        return frame

    def _get_cached_hires_frame(
        self,
        shot_idx: int,
        frame_idx: int,
        target_dims: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Return cached hi-res frame for this shot/frame/size tier if available."""
        size_key = self._size_key_for_dims(target_dims)
        with self._cache_lock:
            return self.hires_frame_buffer.get(shot_idx, {}).get((frame_idx, size_key))

    def _get_compatible_cached_hires_frame(
        self,
        shot_idx: int,
        frame_idx: int,
        target_dims: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Return same-frame cached hi-res from a nearby size tier if exact tier is missing.

        Preference order:
          1) Smallest cached tier that is >= requested dims (best for zooming out slightly)
          2) Largest available tier below requested dims
        """
        target_w, target_h = target_dims
        with self._cache_lock:
            by_shot = self.hires_frame_buffer.get(shot_idx, {})
            same_frame_entries = [
                (size_key, frame)
                for (cached_frame_idx, size_key), frame in by_shot.items()
                if cached_frame_idx == frame_idx
            ]

        if not same_frame_entries:
            return None

        candidates_ge = []
        candidates_lt = []
        for size_key, frame in same_frame_entries:
            w, h = self._dims_for_size_key(size_key)
            area = w * h
            if w >= target_w and h >= target_h:
                over_area = area - (target_w * target_h)
                candidates_ge.append((over_area, area, frame))
            else:
                candidates_lt.append((area, frame))

        if candidates_ge:
            candidates_ge.sort(key=lambda item: (item[0], item[1]))
            return candidates_ge[0][2]

        # If no tier is large enough, use the largest available tier.
        candidates_lt.sort(key=lambda item: item[0], reverse=True)
        return candidates_lt[0][1] if candidates_lt else None

    def _get_nearest_cached_hires_frame(
        self,
        shot_idx: int,
        frame_idx: int,
        target_dims: Tuple[int, int],
        bucket_count: int,
    ) -> Optional[np.ndarray]:
        """Return adjacent cached hi-res frame before dropping to base-res fallback."""
        if bucket_count <= 0 or self.hires_nearest_fallback_radius <= 0:
            return None

        size_key = self._size_key_for_dims(target_dims)
        with self._cache_lock:
            by_shot = self.hires_frame_buffer.get(shot_idx, {})
            for radius in range(1, self.hires_nearest_fallback_radius + 1):
                prev_idx = (frame_idx - radius) % bucket_count
                frame = by_shot.get((prev_idx, size_key))
                if frame is not None:
                    return frame

                next_idx = (frame_idx + radius) % bucket_count
                frame = by_shot.get((next_idx, size_key))
                if frame is not None:
                    return frame
        return None

    def _effective_hires_dims(self, draw_x: int, draw_y: int, draw_w: int, draw_h: int) -> Tuple[int, int]:
        """Clamp zoom decode dimensions with viewport-proportional quality limits."""
        # Target on-screen size with a small overscan so downscaling looks sharper.
        w = min(self.video_width, int(draw_w * self.hires_decode_overscan))
        h = min(self.video_height, int(draw_h * self.hires_decode_overscan))
        w = max(1, w)
        h = max(1, h)

        # Quality budget scales with visible viewport coverage.
        vis_x0 = max(0, draw_x)
        vis_y0 = max(0, draw_y)
        vis_x1 = min(self.window_width, draw_x + draw_w)
        vis_y1 = min(self.window_height, draw_y + draw_h)
        vis_w = max(0, vis_x1 - vis_x0)
        vis_h = max(0, vis_y1 - vis_y0)
        viewport_pixels = max(1, self.window_width * self.window_height)
        coverage = (vis_w * vis_h) / float(viewport_pixels)
        coverage_scale = min(1.0, coverage / max(self.hires_fullres_viewport_threshold, 1e-6))

        source_pixels = self.video_width * self.video_height
        coverage_pixel_budget = max(1, int(source_pixels * coverage_scale))

        pixels = w * h
        pixels = min(pixels, coverage_pixel_budget)
        if pixels > self.hires_decode_max_pixels:
            pixels = self.hires_decode_max_pixels

        target_pixels = max(1, pixels)
        current_pixels = max(1, w * h)
        if current_pixels > target_pixels:
            scale = math.sqrt(target_pixels / float(current_pixels))
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))
        return w, h

    def _clear_hires_cache(self) -> None:
        """Drop zoom-tier cache so stale resolutions do not accumulate across zoom changes."""
        with self._cache_lock:
            self.hires_frame_buffer.clear()
            self.hires_frame_read_order.clear()
            self._visible_hires_requests = []

    def _ordered_visible_shots(self, visible_shots: List[int], visible_weights: Dict[int, int]) -> List[int]:
        """Return visible shots ordered by dominant viewport share first."""
        visible_unique = list(dict.fromkeys(visible_shots))
        visible_unique.sort(key=lambda shot_idx: visible_weights.get(shot_idx, 0), reverse=True)
        return visible_unique

    def _hires_request_depth_for_coverage(self, coverage: float) -> int:
        """Return how many upcoming hi-res buckets to queue based on viewport share."""
        if coverage >= 0.75:
            return min(self.hires_prefetch_max_lead_frames, 8)
        if coverage >= 0.50:
            return min(self.hires_prefetch_max_lead_frames, 6)
        if coverage >= 0.25:
            return min(self.hires_prefetch_max_lead_frames, 4)
        return min(self.hires_prefetch_max_lead_frames, 2)

    def _current_hires_decode_enabled_zoom(self) -> float:
        """Return adaptive zoom threshold for enabling hi-res decode.

        Smaller base tiles (common in smaller windows) switch to hi-res decode earlier
        to avoid long periods of visibly low-quality upscale.
        """
        cell_area = max(1, self.cell_dims[0] * self.cell_dims[1])
        ref_area = max(1, self.hires_decode_reference_cell_pixels)
        area_scale = math.sqrt(min(1.0, cell_area / float(ref_area)))
        threshold = self.hires_decode_enabled_zoom_min + (
            self.hires_decode_enabled_zoom_base - self.hires_decode_enabled_zoom_min
        ) * area_scale
        return max(self.hires_decode_enabled_zoom_min, min(self.hires_decode_enabled_zoom_max, threshold))

    def _subtitle_at(self, abs_ts: float) -> Optional[str]:
        """Return the subtitle text active at *abs_ts* seconds, or None."""
        if not self.subtitles:
            return None
        idx = bisect.bisect_right(self._subtitle_starts, abs_ts) - 1
        if idx < 0:
            return None
        start, end, text = self.subtitles[idx]
        return text if abs_ts <= end else None

    def _get_subtitle_font(self, size: int) -> pygame.font.Font:
        if size not in self._subtitle_font_cache:
            self._subtitle_font_cache[size] = pygame.font.Font(None, size)
        return self._subtitle_font_cache[size]

    def _render_subtitle_on_tile(
        self,
        surface: pygame.Surface,
        text: str,
        draw_x: int,
        draw_y: int,
        draw_w: int,
        draw_h: int,
    ) -> None:
        """Overlay subtitle text as a caption box at the bottom of a tile."""
        visible_tile_h = max(1, min(draw_h, self.window_height))
        zoom_scale = max(1.0, self.grid_zoom / max(self.subtitle_zoom_threshold, 0.1))
        font_size = int(visible_tile_h * 0.065 * min(2.0, zoom_scale))
        font_size = max(18, min(72, font_size))
        cache_key = (text, draw_w, draw_h, font_size)
        overlay = self._subtitle_overlay_cache.get(cache_key)
        if overlay is not None:
            overlay_y = draw_y + draw_h - overlay.get_height() - max(3, draw_h // 40)
            overlay_y = max(draw_y, overlay_y)
            surface.blit(overlay, (draw_x, overlay_y))
            return

        font = self._get_subtitle_font(font_size)
        h_pad = max(4, draw_w // 24)
        v_pad = max(3, draw_h // 40)
        max_line_w = draw_w - h_pad * 2

        # Word-wrap each raw line independently.
        wrapped: List[str] = []
        for raw_line in text.split("\n"):
            words = raw_line.split()
            if not words:
                wrapped.append("")
                continue
            current: List[str] = []
            for word in words:
                candidate = " ".join(current + [word])
                if font.size(candidate)[0] <= max_line_w:
                    current.append(word)
                else:
                    if current:
                        wrapped.append(" ".join(current))
                    current = [word]
            if current:
                wrapped.append(" ".join(current))

        if not wrapped:
            return

        line_h = font.get_linesize()
        total_text_h = line_h * len(wrapped)
        box_h = total_text_h + v_pad * 2
        overlay = pygame.Surface((draw_w, box_h), pygame.SRCALPHA)
        bg = pygame.Surface((draw_w - h_pad * 2, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        overlay.blit(bg, (h_pad, 0))

        for i, line in enumerate(wrapped):
            text_surf = font.render(line, True, (255, 255, 255))
            tx = h_pad + (draw_w - h_pad * 2 - text_surf.get_width()) // 2
            ty = v_pad + i * line_h
            overlay.blit(text_surf, (tx, ty))

        if len(self._subtitle_overlay_cache) >= self._subtitle_overlay_cache_max_entries:
            self._subtitle_overlay_cache.pop(next(iter(self._subtitle_overlay_cache)))
        self._subtitle_overlay_cache[cache_key] = overlay

        overlay_y = draw_y + draw_h - overlay.get_height() - v_pad
        overlay_y = max(draw_y, overlay_y)
        surface.blit(overlay, (draw_x, overlay_y))

    def _hires_budget_for_visibility(self, request_count: int, visible_weights: Dict[int, int]) -> int:
        """Dynamically raise hi-res budget when one tile dominates the viewport.

        The boost is a continuous multiplicative function of three factors:
          - dominant_coverage  : how much of the viewport the largest tile occupies
          - headroom_factor    : fraction of per-frame render budget still available
          - pressure_factor    : 1 minus the fraction of frame budget spent on decoding

        This means the boost scales down smoothly as the system gets busier, rather
        than jumping between fixed subtract thresholds.
        """
        if request_count <= 0:
            return 0

        viewport_pixels = max(1, self.window_width * self.window_height)
        dominant_pixels = max(visible_weights.values(), default=0)
        dominant_coverage = dominant_pixels / float(viewport_pixels)

        # Coverage-based base boost (kick in at 60%).
        if dominant_coverage >= 0.85:
            base_boost = 8
        elif dominant_coverage >= 0.72:
            base_boost = 6
        elif dominant_coverage >= 0.60:
            base_boost = 3
        else:
            base_boost = 0

        target_frame_ms = 1000.0 / max(float(self.fps), 1.0)

        # Render-headroom factor: how much of the frame budget is still free.
        render_samples = list(self.render_times)[-20:]
        avg_render_ms = sum(render_samples) / len(render_samples) if render_samples else 0.0
        headroom_factor = max(0.0, min(1.0, (target_frame_ms - avg_render_ms) / target_frame_ms))

        # Decode-pressure factor: high decode load reduces willingness to queue more.
        decode_samples = list(self.decode_times)[-20:]
        avg_decode_ms = sum(decode_samples) / len(decode_samples) if decode_samples else 0.0
        pressure_factor = max(0.0, min(1.0, 1.0 - (avg_decode_ms / max(target_frame_ms, 1.0))))

        # When only a handful of tiles are visible, spend more decode budget to
        # replace low-res placeholders quickly.
        visible_count = max(1, len(visible_weights))
        if visible_count <= 2:
            small_set_boost = 18
        elif visible_count <= 4:
            small_set_boost = 12
        elif visible_count <= 6:
            small_set_boost = 6
        else:
            small_set_boost = 0

        final_boost = int(base_boost * headroom_factor * pressure_factor)
        if small_set_boost > 0:
            small_set_factor = (0.45 + 0.55 * headroom_factor) * max(0.35, pressure_factor)
            final_boost += int(small_set_boost * small_set_factor)
        # Guarantee at least a token boost of 1 when we are in the coverage zone,
        # so we do not stall hi-res prefetch entirely during a brief spike.
        if dominant_coverage >= 0.60 and final_boost == 0:
            final_boost = 1

        budget = self.hires_prefetch_budget_per_cycle + final_boost
        budget = min(self.hires_prefetch_budget_max_boosted, budget)
        budget = min(request_count, budget)
        return max(1, budget)

    def _prune_invisible_caches(self, visible_shot_indices: List[int]) -> None:
        """Drop off-screen hi-res caches so zoomed-in playback prioritizes visible content."""
        if self.grid_zoom < self._current_hires_decode_enabled_zoom():
            return

        visible = set(visible_shot_indices)
        with self._cache_lock:
            for shot_idx in list(self.hires_frame_buffer.keys()):
                if shot_idx in visible:
                    continue
                self.hires_frame_buffer.pop(shot_idx, None)
                self.hires_frame_read_order.pop(shot_idx, None)

    def _maybe_prune_hires_cache_for_zoom(self, old_zoom: float, new_zoom: float) -> None:
        """Prune hi-res cache when zoom tier changes enough to invalidate cached size buckets."""
        # Keep hi-res cache when dipping below threshold so rapid zoom-in/out
        # does not drop back to lowest-quality placeholders.
        if new_zoom < self._current_hires_decode_enabled_zoom():
            return
        if old_zoom <= 0:
            return
        zoom_ratio = max(new_zoom / old_zoom, old_zoom / new_zoom)
        if zoom_ratio >= self.hires_cache_prune_zoom_ratio:
            self._clear_hires_cache()

    def get_shot_frame(
        self,
        shot_idx: int,
        time_offset: float,
        frame_count: int,
        target_dims: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get a frame from a shot, using cached frames where possible.
        Avoids expensive seeking by caching actual video frames.
        
        Parameters
        ----------
        shot_idx : int
            Index of the shot (0-based)
        time_offset : float
            Time offset within the shot in seconds
        
        Returns
        -------
        Optional[np.ndarray]
            Resized frame as (height, width, 3) RGB array, or None if unavailable
        """
        if shot_idx >= len(self.shot_timecodes):
            return None
        
        start_time, end_time = self.shot_timecodes[shot_idx]
        shot_duration = end_time - start_time
        if shot_duration <= 0:
            return None
        
        bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
        frame_idx = int(time_offset / self.cache_time_quantum) % bucket_count
        frame_time = self._frame_idx_to_time(shot_idx, frame_idx)

        if target_dims is None:
            target_dims = self.cell_dims

        # Zoomed-in path: use cached hi-res frame if available.
        if target_dims[0] > self.cell_dims[0] or target_dims[1] > self.cell_dims[1]:
            hires_frame_idx = self._frame_idx_for_quantum(shot_idx, time_offset, self.hires_time_quantum)
            hires_bucket_count = max(1, int(shot_duration / self.hires_time_quantum) + 1)
            frame = self._get_cached_hires_frame(shot_idx, hires_frame_idx, target_dims)
            if frame is not None:
                self.cache_hits += 1
                self.shot_stats["exact_hits"][shot_idx] += 1
                self.shot_stats["exact_hit_frames"][shot_idx].append(frame_count)
                self._record_shot_lag(shot_idx, frame_idx, frame_idx)
                return frame
            frame = self._get_compatible_cached_hires_frame(shot_idx, hires_frame_idx, target_dims)
            if frame is not None:
                self.cache_fallback_reuses += 1
                self.shot_stats["fallback_hits"][shot_idx] += 1
                self._record_shot_lag(shot_idx, frame_idx, frame_idx)
                return frame
            frame = self._get_nearest_cached_hires_frame(
                shot_idx,
                hires_frame_idx,
                target_dims,
                hires_bucket_count,
            )
            if frame is not None:
                self.cache_fallback_reuses += 1
                self.shot_stats["fallback_hits"][shot_idx] += 1
                self._record_shot_lag(shot_idx, frame_idx, frame_idx)
                return frame
        
        # Check if frame is in buffer (cache hit)
        with self._cache_lock:
            if shot_idx in self.frame_buffer and frame_idx in self.frame_buffer[shot_idx]:
                self.cache_hits += 1
                self.shot_stats["exact_hits"][shot_idx] += 1
                self.shot_stats["exact_hit_frames"][shot_idx].append(frame_count)
                self._record_shot_lag(shot_idx, frame_idx, frame_idx)
                return self.frame_buffer[shot_idx][frame_idx]
        
        # Frame not ready in queue: never decode in render loop.
        self.cache_misses += 1
        with self._cache_lock:
            if shot_idx in self.latest_ready_frame:
                self.cache_fallback_reuses += 1
                self.shot_stats["fallback_hits"][shot_idx] += 1
                self._record_shot_lag(shot_idx, frame_idx, self.latest_ready_idx.get(shot_idx))
                return self.latest_ready_frame[shot_idx]

        self.decode_misses += 1
        self.queue_starvations += 1
        self.shot_stats["starvations"][shot_idx] += 1
        return None
    
    def _get_or_decode_frame(
        self,
        shot_idx: int,
        frame_idx: int,
        frame_time: float,
        source_video=None,
        target_dims: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Decode a single frame from the video and resize it.
        
        Parameters
        ----------
        shot_idx : int
            Shot index (for context)
        frame_idx : int
            Frame index within shot
        frame_time : float
            Absolute time in video to extract frame from
        
        Returns
        -------
        Optional[np.ndarray]
            Resized frame, or None if decode fails
        """
        try:
            video = source_video or self.video
            frame = video.get_frame(frame_time)
            if frame is None:
                return None
            
            # Resize to requested output dimensions.
            if target_dims is None:
                target_dims = self.cell_dims
            cell_w, cell_h = target_dims
            if frame.shape != (cell_h, cell_w, 3):
                if HAS_CV2 and cv2 is not None:
                    frame = cv2.resize(frame.astype('uint8'), (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                else:
                    img = Image.fromarray(frame.astype('uint8'))
                    img = img.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
                    frame = np.array(img)
            
            return frame.astype(np.uint8)
        except Exception:
            return None

    def _log_shot_metrics(self, frame_count: int, final: bool = False):
        """Print compact per-shot smoothness diagnostics for the worst shots."""
        if not final and frame_count % self.shot_qos_log_every_n_frames != 0:
            return

        rows = []
        window_frames = max(1, int(self.shot_qos_fresh_window_seconds * self.fps))
        min_frame_in_window = max(0, frame_count - window_frames)
        for shot_idx in range(len(self.shot_timecodes)):
            exact = self.shot_stats["exact_hits"][shot_idx]
            fallback = self.shot_stats["fallback_hits"][shot_idx]
            starve = self.shot_stats["starvations"][shot_idx]
            accesses = exact + fallback + starve
            if accesses == 0:
                continue

            exact_rate = exact / accesses * 100
            fallback_rate = fallback / accesses * 100
            lag_samples = self.shot_stats["lag_samples"][shot_idx]
            avg_lag = (self.shot_stats["lag_sum"][shot_idx] / lag_samples) if lag_samples else 0.0
            max_lag = self.shot_stats["lag_max"][shot_idx]
            avg_lag_s = avg_lag * self.cache_time_quantum
            max_lag_s = max_lag * self.cache_time_quantum
            dec_samples = self.shot_stats["decode_samples"][shot_idx]
            avg_decode_ms = (self.shot_stats["decode_ms_sum"][shot_idx] / dec_samples) if dec_samples else 0.0

            exact_hit_frames = self.shot_stats["exact_hit_frames"][shot_idx]
            while exact_hit_frames and exact_hit_frames[0] < min_frame_in_window:
                exact_hit_frames.popleft()
            fresh_fps = len(exact_hit_frames) / max(self.shot_qos_fresh_window_seconds, 1)

            # Higher fallback and lag indicate lower perceptual smoothness.
            score = fallback_rate + avg_lag_s * 30.0 + starve * 5.0
            rows.append((
                score,
                shot_idx,
                exact_rate,
                fallback_rate,
                avg_lag,
                max_lag,
                avg_lag_s,
                max_lag_s,
                starve,
                avg_decode_ms,
                fresh_fps,
            ))

        if not rows:
            return

        rows.sort(reverse=True)
        top_n = 8 if final else 5
        signature = tuple(
            (r[1], round(r[3], 1), round(r[6], 2), round(r[10], 1))
            for r in rows[:top_n]
        )
        if not final and signature == self.shot_qos_last_signature:
            return
        self.shot_qos_last_signature = signature

        summary = []
        for (
            _,
            shot_idx,
            exact_rate,
            fallback_rate,
            avg_lag,
            max_lag,
            avg_lag_s,
            max_lag_s,
            starve,
            avg_decode_ms,
            fresh_fps,
        ) in rows[:top_n]:
            summary.append(
                f"S{shot_idx + 1}: fresh {fresh_fps:.1f}fps hit {exact_rate:.0f}% fb {fallback_rate:.0f}% "
                f"lag {avg_lag_s:.2f}/{max_lag_s:.2f}s ({avg_lag:.1f}/{max_lag}b)"
                f" starve {starve} dec {avg_decode_ms:.1f}ms"
            )

        print(f"[Frame {frame_count}] Shot QoS worst: " + " || ".join(summary), flush=True)
    
    def render_frame(self, frame_count: int) -> pygame.Surface:
        """
        Render the current grid frame with optimizations.
        
        Returns
        -------
        pygame.Surface
            Rendered frame as a Pygame surface
        """
        render_start = time.perf_counter()
        
        # Create surface
        surface = pygame.Surface((self.window_width, self.window_height))
        surface.fill((0, 0, 0))  # Black background
        
        rows, cols = self.layout
        cell_w, cell_h = self.cell_dims
        selected_order = {
            shot_idx: order + 1
            for order, shot_idx in enumerate(self.selected_shot_indices)
        }
        badge_font = pygame.font.Font(None, 24) if selected_order else None
        visible_hires_requests: List[Tuple[int, int, Tuple[int, int]]] = []
        visible_shot_indices: List[int] = []
        visible_shot_weights: Dict[int, int] = {}
        subtitle_candidates: List[Tuple[int, float, int, int, int, int]] = []
        viewport_pixels = max(1, self.window_width * self.window_height)
        hovered_shot_idx = self._hovered_shot_index()
        
        # Render each shot with culling (skip off-screen cells)
        for shot_idx, (start_time, end_time) in enumerate(self.shot_timecodes):
            shot_duration = end_time - start_time
            if shot_duration <= 0:
                continue
            
            # Calculate position
            row = shot_idx // cols
            col = shot_idx % cols
            x = self.pad_x + col * cell_w
            y = self.pad_y + row * cell_h

            draw_x, draw_y, draw_w, draw_h = self._apply_grid_transform(x, y, cell_w, cell_h)

            if draw_w <= 0 or draw_h <= 0:
                continue
            
            # Culling: Skip cells that are completely off-screen
            if (
                draw_x + draw_w < 0
                or draw_x >= self.window_width
                or draw_y + draw_h < 0
                or draw_y >= self.window_height
            ):
                continue

            vis_x0 = max(0, draw_x)
            vis_y0 = max(0, draw_y)
            vis_x1 = min(self.window_width, draw_x + draw_w)
            vis_y1 = min(self.window_height, draw_y + draw_h)
            visible_pixels = max(0, vis_x1 - vis_x0) * max(0, vis_y1 - vis_y0)

            visible_shot_indices.append(shot_idx)
            weight = visible_pixels
            if hovered_shot_idx is not None and shot_idx == hovered_shot_idx:
                # Cursor-hovered shot gets priority so upcoming zoom has warm hi-res cache.
                weight += int(viewport_pixels * self.hires_cursor_hint_weight_boost)
            visible_shot_weights[shot_idx] = weight
            
            # Calculate playback position within this shot (looping)
            # Each shot loops independently as global time advances
            time_in_shot = self.current_time % shot_duration
            hires_frame_idx = self._frame_idx_for_quantum(shot_idx, time_in_shot, self.hires_time_quantum)

            target_dims: Tuple[int, int] = self.cell_dims
            hires_zoom_threshold = self._current_hires_decode_enabled_zoom()
            if self.grid_zoom >= hires_zoom_threshold:
                target_dims = self._effective_hires_dims(draw_x, draw_y, draw_w, draw_h)
                if target_dims[0] > self.cell_dims[0] or target_dims[1] > self.cell_dims[1]:
                    coverage = visible_pixels / float(viewport_pixels)
                    lead_depth = self._hires_request_depth_for_coverage(coverage)
                    hires_bucket_count = max(1, int(shot_duration / self.hires_time_quantum) + 1)
                    for offset in range(lead_depth):
                        req_idx = (hires_frame_idx + offset) % hires_bucket_count
                        visible_hires_requests.append((shot_idx, req_idx, target_dims))
            elif hovered_shot_idx is not None and shot_idx == hovered_shot_idx:
                # Before hitting the zoom threshold, pre-decode a few hi-res frames for
                # the hovered tile at the threshold-equivalent size.
                threshold_scale = hires_zoom_threshold / max(self.grid_zoom, 1e-6)
                hinted_draw_w = max(draw_w, int(draw_w * threshold_scale))
                hinted_draw_h = max(draw_h, int(draw_h * threshold_scale))
                hinted_dims = self._effective_hires_dims(draw_x, draw_y, hinted_draw_w, hinted_draw_h)
                if hinted_dims[0] > self.cell_dims[0] or hinted_dims[1] > self.cell_dims[1]:
                    hires_bucket_count = max(1, int(shot_duration / self.hires_time_quantum) + 1)
                    lead_depth = min(self.hires_prefetch_max_lead_frames, self.hires_cursor_hint_lead_frames)
                    for offset in range(lead_depth):
                        req_idx = (hires_frame_idx + offset) % hires_bucket_count
                        visible_hires_requests.append((shot_idx, req_idx, hinted_dims))
            
            # Get frame (will use cache if available)
            frame = self.get_shot_frame(shot_idx, time_in_shot, frame_count, target_dims=target_dims)
            if frame is None:
                continue
            
            # Convert frame to Pygame surface (optimized)
            frame_uint8 = frame if frame.dtype == np.uint8 else frame.astype(np.uint8)
            
            # Use faster numpy transpose for RGB->BGR for pygame
            try:
                # Create pygame surface from numpy array
                frame_surface = pygame.surfarray.make_surface(
                    np.transpose(frame_uint8, (1, 0, 2))
                )
                if frame_surface.get_width() != draw_w or frame_surface.get_height() != draw_h:
                    frame_surface = pygame.transform.smoothscale(frame_surface, (draw_w, draw_h))
                # Blit to surface
                surface.blit(frame_surface, (draw_x, draw_y))
                if shot_idx in selected_order and badge_font is not None:
                    border_rect = pygame.Rect(draw_x, draw_y, draw_w, draw_h)
                    pygame.draw.rect(surface, (245, 205, 70), border_rect, width=4, border_radius=4)
                    badge_rect = pygame.Rect(draw_x + 6, draw_y + 6, 24, 24)
                    pygame.draw.rect(surface, (245, 205, 70), badge_rect, border_radius=12)
                    badge_text = badge_font.render(str(selected_order[shot_idx]), True, (20, 20, 20))
                    surface.blit(
                        badge_text,
                        (
                            badge_rect.x + (badge_rect.width - badge_text.get_width()) // 2,
                            badge_rect.y + (badge_rect.height - badge_text.get_height()) // 2,
                        ),
                    )
                if self.subtitles and self.subtitles_visible and self.grid_zoom >= self.subtitle_zoom_threshold:
                    abs_ts = start_time + time_in_shot
                    subtitle_candidates.append((visible_pixels, abs_ts, draw_x, draw_y, draw_w, draw_h))
            except Exception as e:
                # Silently skip if blit fails
                pass

        if subtitle_candidates:
            for _visible_pixels, abs_ts, draw_x, draw_y, draw_w, draw_h in subtitle_candidates:
                sub_text = self._subtitle_at(abs_ts)
                if sub_text:
                    self._render_subtitle_on_tile(surface, sub_text, draw_x, draw_y, draw_w, draw_h)

        with self._cache_lock:
            self._visible_hires_requests = visible_hires_requests
            self._visible_shot_indices = visible_shot_indices
            self._visible_shot_weights = visible_shot_weights
        self._prune_invisible_caches(visible_shot_indices)
        
        render_time = (time.perf_counter() - render_start) * 1000  # Convert to ms
        self.render_times.append(render_time)
        return surface
    
    def _render_warmup_screen(self) -> None:
        """Draw a progress bar overlay while frame caches are warming up."""
        sw, sh = self.window_width, self.window_height
        self.display.fill((12, 12, 12))

        font_large = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)

        title = font_large.render("Warming up…", True, (200, 200, 200))
        self.display.blit(title, (sw // 2 - title.get_width() // 2, sh // 2 - 60))

        bar_w = int(sw * 0.6)
        bar_h = 18
        bar_x = (sw - bar_w) // 2
        bar_y = sh // 2 - 10

        total = max(self._warmup_total, 1)
        progress = self._warmup_progress
        fill_w = int(bar_w * min(progress / total, 1.0))

        pygame.draw.rect(self.display, (45, 45, 45), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        if fill_w > 0:
            pygame.draw.rect(self.display, (60, 130, 230), (bar_x, bar_y, fill_w, bar_h), border_radius=6)
        pygame.draw.rect(self.display, (90, 90, 90), (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=6)

        pct = int(progress / total * 100)
        lbl = font_small.render(f"{progress} / {total} frames  ({pct}%)", True, (150, 150, 150))
        self.display.blit(lbl, (sw // 2 - lbl.get_width() // 2, bar_y + bar_h + 10))

        pygame.display.flip()

    def _render_resize_screen(self, pending_size: Tuple[int, int]) -> None:
        """Draw a lightweight screen while waiting for resize to settle."""
        sw, sh = self.window_width, self.window_height
        self.display.fill((10, 10, 10))
        font_large = pygame.font.Font(None, 42)
        font_small = pygame.font.Font(None, 26)

        title = font_large.render("Resizing window...", True, (210, 210, 210))
        detail = font_small.render(
            f"Release to apply: {pending_size[0]}x{pending_size[1]}",
            True,
            (160, 160, 160),
        )
        self.display.blit(title, (sw // 2 - title.get_width() // 2, sh // 2 - 26))
        self.display.blit(detail, (sw // 2 - detail.get_width() // 2, sh // 2 + 14))

    def _rewarm_caches_with_progress(self) -> bool:
        """Run warmup and progress UI; returns False if user requested exit."""
        self._warmup_progress = 0
        self._warmup_total = 0
        self._warmup_done = False
        warmup_thread = threading.Thread(target=self._warmup_frame_caches, daemon=True)
        warmup_thread.start()

        while not self._warmup_done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    warmup_thread.join(timeout=1.0)
                    return False
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                    warmup_thread.join(timeout=1.0)
                    return False
                if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self._resize_pending_size = (event.w, event.h)
                    self._resize_last_event_ts = time.perf_counter()
                elif event.type == pygame.WINDOWSIZECHANGED and not self.fullscreen:
                    self._resize_pending_size = (event.x, event.y)
                    self._resize_last_event_ts = time.perf_counter()
            self._render_warmup_screen()
            self.clock.tick(30)

        warmup_thread.join()
        return True

    def run(self):
        """Run the interactive grid viewer."""
        if self.verbose:
            print(f"Grid Viewer: {len(self.shot_timecodes)} shots, {self.layout[0]}×{self.layout[1]} grid")
            print(f"Cell size: {self.cell_dims[0]}×{self.cell_dims[1]}")
            print(f"Target FPS: {self.fps}")
            print(f"Controls: SPACE=pause/play, LEFT/RIGHT=seek, mouse wheel or +/- to zoom, 0=reset zoom, Q=quit")

        # Warm up caches on a background thread while showing a progress bar
        if self.verbose:
            print("Warming up frame caches...")
        if not self._rewarm_caches_with_progress():
            pygame.quit()
            return

        # Start background prefetching thread
        self._start_prefetch_thread()

        running = True
        frame_count = 0
        loop_start_time = time.perf_counter()
        
        while running:
            frame_loop_start = time.perf_counter()
            active_viewer = self._sequence_viewer or self._shot_viewer
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self._resize_pending_size = (event.w, event.h)
                    self._resize_last_event_ts = time.perf_counter()
                elif event.type == pygame.WINDOWSIZECHANGED and not self.fullscreen:
                    self._resize_pending_size = (event.x, event.y)
                    self._resize_last_event_ts = time.perf_counter()
                elif event.type == pygame.MOUSEMOTION:
                    if self._context_menu is not None:
                        self._context_menu.update_hover(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4 and active_viewer is None:  # wheel up
                        self._zoom_grid(1.12, event.pos)
                    elif event.button == 5 and active_viewer is None:  # wheel down
                        self._zoom_grid(1.0 / 1.12, event.pos)
                    elif event.button == 3:  # right-click
                        if active_viewer is not None:
                            continue
                        self._context_menu = None
                        shot_idx = self._hovered_shot_index(event.pos)
                        if shot_idx is not None:
                            if shot_idx not in self.selected_shot_indices:
                                self.selected_shot_indices = [shot_idx]

                            if len(self.selected_shot_indices) > 1:
                                menu_label = f"View Selected ({len(self.selected_shot_indices)})"
                                payload = {
                                    "kind": "sequence",
                                    "shot_indices": self.selected_shot_indices[:],
                                    "start_position": self.selected_shot_indices.index(shot_idx),
                                }
                                menu_items = [
                                    menu_label,
                                    "Export clips (no re-encode)",
                                    "Export clips (frame-accurate)",
                                    "Export sequence",
                                ]
                            else:
                                menu_label = "View"
                                payload = {
                                    "kind": "shot",
                                    "shot_idx": shot_idx,
                                }
                                menu_items = [
                                    menu_label,
                                    "Export clips (no re-encode)",
                                    "Export clips (frame-accurate)",
                                ]

                            menu = ContextMenu(
                                event.pos[0],
                                event.pos[1],
                                menu_items,
                                payload,
                            )
                            menu.clamp_to_screen(self.window_width, self.window_height)
                            self._context_menu = menu
                    elif event.button == 1:  # left-click
                        if self._context_menu is not None:
                            item_idx = self._context_menu.item_at(event.pos)
                            if item_idx is None:
                                # Click outside menu — dismiss
                                self._context_menu = None
                            else:
                                payload = self._context_menu.payload
                                selected_item = self._context_menu.items[item_idx]
                                self._context_menu = None
                                if selected_item.startswith("View"):
                                    if payload["kind"] == "sequence":
                                        shot_indices = cast(List[int], payload["shot_indices"])
                                        start_position = cast(int, payload["start_position"])
                                        self._open_sequence_viewer(
                                            shot_indices,
                                            start_position,
                                        )
                                    else:
                                        shot_idx = cast(int, payload["shot_idx"])
                                        self._open_shot_viewer(shot_idx)
                                elif selected_item == "Export clips (no re-encode)":
                                    if payload["kind"] == "sequence":
                                        shot_indices = cast(List[int], payload["shot_indices"])
                                    else:
                                        shot_indices = [cast(int, payload["shot_idx"])]
                                    self._export_clips(shot_indices)
                                elif selected_item == "Export clips (frame-accurate)":
                                    if payload["kind"] == "sequence":
                                        shot_indices = cast(List[int], payload["shot_indices"])
                                    else:
                                        shot_indices = [cast(int, payload["shot_idx"])]
                                    self._export_clips_frame_accurate(shot_indices)
                                elif selected_item == "Export sequence" and payload["kind"] == "sequence":
                                    shot_indices = cast(List[int], payload["shot_indices"])
                                    self._export_sequence(shot_indices)
                        elif self._sequence_viewer is not None:
                            sv = self._sequence_viewer
                            if sv.close_btn_rect.collidepoint(event.pos):
                                self._close_active_viewer()
                            elif sv.prev_btn_rect.collidepoint(event.pos) and sv.can_go_prev():
                                sv.step(-1)
                            elif sv.next_btn_rect.collidepoint(event.pos) and sv.can_go_next():
                                sv.step(1)
                        elif self._shot_viewer is not None:
                            sv = self._shot_viewer
                            if sv.close_btn_rect.collidepoint(event.pos):
                                self._close_active_viewer()
                            elif sv.prev_btn_rect.collidepoint(event.pos) and sv.shot_idx > 0:
                                self._open_shot_viewer(sv.shot_idx - 1)
                            elif sv.next_btn_rect.collidepoint(event.pos) and sv.shot_idx < sv.total_shots - 1:
                                self._open_shot_viewer(sv.shot_idx + 1)
                        else:
                            self._context_menu = None
                            shot_idx = self._hovered_shot_index(event.pos)
                            if shot_idx is None:
                                self.selected_shot_indices.clear()
                                self._last_left_click_shot_idx = None
                            else:
                                now = time.perf_counter()
                                is_double_click = (
                                    getattr(event, "clicks", 1) >= 2
                                    or (
                                        self._last_left_click_shot_idx == shot_idx
                                        and (now - self._last_left_click_ts) <= self._double_click_seconds
                                    )
                                )
                                self._last_left_click_ts = now
                                self._last_left_click_shot_idx = shot_idx
                                if is_double_click:
                                    self._snap_shot_to_window(shot_idx)
                                else:
                                    self._toggle_selected_shot(shot_idx)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if active_viewer is not None:
                            self._close_active_viewer()
                        elif self._context_menu is not None:
                            self._context_menu = None
                        else:
                            running = False
                    elif event.key == pygame.K_q:
                        running = False
                    elif self._sequence_viewer is not None:
                        sv = self._sequence_viewer
                        if event.key == pygame.K_LEFT and sv.can_go_prev():
                            sv.step(-1)
                        elif event.key == pygame.K_RIGHT and sv.can_go_next():
                            sv.step(1)
                    elif self._shot_viewer is not None:
                        sv = self._shot_viewer
                        if event.key == pygame.K_LEFT and sv.shot_idx > 0:
                            self._open_shot_viewer(sv.shot_idx - 1)
                        elif event.key == pygame.K_RIGHT and sv.shot_idx < sv.total_shots - 1:
                            self._open_shot_viewer(sv.shot_idx + 1)
                    elif event.key == pygame.K_s:
                        self.subtitles_visible = not self.subtitles_visible
                    elif event.key == pygame.K_BACKSPACE:
                        self.selected_shot_indices.clear()
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self._zoom_grid(1.12)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self._zoom_grid(1.0 / 1.12)
                    elif event.key in (pygame.K_0, pygame.K_KP0):
                        self._reset_grid_zoom()
                    elif event.key == pygame.K_LEFT and self._can_pan_grid():
                        self._pan_grid(self.pan_pixels_per_second / self.fps, 0.0)
                    elif event.key == pygame.K_RIGHT and self._can_pan_grid():
                        self._pan_grid(-self.pan_pixels_per_second / self.fps, 0.0)
                    elif event.key == pygame.K_UP and self._can_pan_grid():
                        self._pan_grid(0.0, self.pan_pixels_per_second / self.fps)
                    elif event.key == pygame.K_DOWN and self._can_pan_grid():
                        self._pan_grid(0.0, -self.pan_pixels_per_second / self.fps)
                    elif event.key == pygame.K_LEFT:
                        self.current_time = max(0, self.current_time - 1.0)
                    elif event.key == pygame.K_RIGHT:
                        self.current_time = min(self.max_duration, self.current_time + 1.0)

            # While the user is actively resizing, pause normal rendering and wait
            # for dimensions to settle before re-applying layout and warming caches.
            if self._resize_pending_size is not None:
                if time.perf_counter() - self._resize_last_event_ts < self._resize_settle_seconds:
                    self._render_resize_screen(self._resize_pending_size)
                    pygame.display.flip()
                    self.clock.tick(60)
                    continue

                target_w, target_h = self._resize_pending_size
                self._resize_pending_size = None
                self._context_menu = None
                resized = self._handle_window_resize(target_w, target_h)
                if resized:
                    self._stop_prefetch_thread()
                    if not self._rewarm_caches_with_progress():
                        running = False
                        continue
                    self._start_prefetch_thread()
                continue

            # Continuous panning from keyboard hold and edge-hover
            if active_viewer is None:
                dt = 1.0 / self.fps
                self._update_keyboard_pan(dt)
                self._update_edge_pan(dt)
            
            # Update playback time
            if self.playing:
                self.current_time += 1.0 / self.fps
                if self.current_time >= self.max_duration:
                    self.current_time = 0.0  # Loop
            
            # Render and display
            frame_surface = self.render_frame(frame_count)
            self.display.blit(frame_surface, (0, 0))

            # Overlay viewer (drawn before HUD so HUD stays on top)
            if self._sequence_viewer is not None:
                self._sequence_viewer.update()
                self._sequence_viewer.draw(self.display)
            elif self._shot_viewer is not None:
                self._shot_viewer.update()
                self._shot_viewer.draw(self.display)

            # Context menu (topmost layer)
            if self._context_menu is not None:
                self._context_menu.draw(self.display)
            
            # Display HUD
            self._render_hud()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
            
            frame_count += 1
            
            # Track frame time
            frame_time = (time.perf_counter() - frame_loop_start) * 1000  # Convert to ms
            self.frame_times.append(frame_time)
            
            # Log performance metrics periodically (every N frames or on first frame)
            if self.verbose and (frame_count == 1 or frame_count % self.perf_log_interval == 0):
                self._log_performance(frame_count)
        
        # Stop background thread and cleanup
        self._stop_prefetch_thread()
        self._close_active_viewer()
        pygame.quit()
        self.video.close()
        
        # Final performance summary
        if self.verbose:
            self._log_performance(frame_count, final=True)
    
    def _log_performance(self, frame_count: int, final: bool = False):
        """Log performance metrics including FPS and cache stats."""
        if len(self.frame_times) == 0:
            return
        
        frame_times_list = list(self.frame_times)
        avg_frame_time = sum(frame_times_list) / len(frame_times_list)
        actual_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        min_frame_time = min(frame_times_list)
        max_frame_time = max(frame_times_list)
        
        # Calculate cache stats
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_accesses * 100) if total_accesses > 0 else 0
        fallback_rate = (self.cache_fallback_reuses / total_accesses * 100) if total_accesses > 0 else 0

        # Queue depth snapshot
        queue_depths = []
        with self._cache_lock:
            for shot_idx, (start_time, end_time) in enumerate(self.shot_timecodes):
                shot_duration = end_time - start_time
                if shot_duration <= 0:
                    continue
                bucket_count = self.shot_bucket_counts.get(shot_idx, 1)
                playback_idx = int((self.current_time % shot_duration) / self.cache_time_quantum) % bucket_count
                needed = [(playback_idx + offset) % bucket_count for offset in range(self.prefetch_base_depth)]
                ready = self.frame_buffer.get(shot_idx, {})
                depth = sum(1 for idx in needed if idx in ready)
                queue_depths.append(depth)
        avg_queue_depth = (sum(queue_depths) / len(queue_depths)) if queue_depths else 0
        
        # Calculate decode stats
        avg_decode_time = (sum(self.decode_times) / len(self.decode_times)) if self.decode_times else 0
        
        # Calculate render stats
        avg_render_time = (sum(self.render_times) / len(self.render_times)) if self.render_times else 0
        
        # Build log message
        log_msg = f"[Frame {frame_count}] "
        log_msg += f"FPS: {actual_fps:.1f} (target {self.fps}) | "
        log_msg += f"Frame time: {avg_frame_time:.2f}ms (min {min_frame_time:.2f}ms, max {max_frame_time:.2f}ms) | "
        log_msg += (
            f"Cache: {hit_rate:.1f}% exact hits ({self.cache_hits} hits, {self.cache_misses} misses, "
            f"{fallback_rate:.1f}% fallback reuse, {self.decode_misses} decode misses) | "
        )
        log_msg += f"Queue: {avg_queue_depth:.1f}/{self.prefetch_base_depth} avg depth ({self.queue_starvations} starvations) | "
        log_msg += f"Render: {avg_render_time:.2f}ms | "
        log_msg += f"Decode: {avg_decode_time:.2f}ms"
        
        if final:
            log_msg += " [FINAL]"
        
        print(log_msg, flush=True)
        self._log_shot_metrics(frame_count, final=final)
        
        # Print playback position on final report
        if final:
            print(f"Playback position: {self.current_time:.1f}s / {self.max_duration:.1f}s", flush=True)

    def _open_shot_viewer(self, shot_idx: int) -> None:
        """Open the in-window shot viewer for shot_idx (0-based)."""
        self._close_active_viewer()
        shot_idx = max(0, min(shot_idx, len(self.shot_timecodes) - 1))
        start_time, end_time = self.shot_timecodes[shot_idx]
        self._shot_viewer = ShotViewer(
            shot_idx=shot_idx,
            start_time=start_time,
            end_time=end_time,
            video_path=self.video_path,
            fps=self.fps,
            screen_w=self.window_width,
            screen_h=self.window_height,
            total_shots=len(self.shot_timecodes),
        )
        if self.verbose:
            print(f"[ShotViewer] Opening shot {shot_idx + 1} ({start_time:.2f}s – {end_time:.2f}s)", flush=True)

    def _choose_export_directory(self) -> Optional[Path]:
        """Open a directory picker and return selected output folder.

        Linux-first strategy:
        1) Use native dialogs (zenity/kdialog) to avoid toolkit conflicts with Pygame.
        2) Fall back to Tk only if native tools are unavailable.
        """
        initial_dir = str(self.video_path.parent)

        zenity_path = shutil.which("zenity")
        if zenity_path is not None:
            cmd = [
                zenity_path,
                "--file-selection",
                "--directory",
                "--title=Select directory for exported clips",
                f"--filename={initial_dir}{os.sep}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                selected = result.stdout.strip()
                return Path(selected) if selected else None
            if result.returncode not in (1,):
                err = result.stderr.strip().splitlines()
                tail = err[-1] if err else "unknown zenity error"
                print(f"[Export clips] zenity failed: {tail}", flush=True)

        kdialog_path = shutil.which("kdialog")
        if kdialog_path is not None:
            cmd = [
                kdialog_path,
                "--getexistingdirectory",
                initial_dir,
                "--title",
                "Select directory for exported clips",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                selected = result.stdout.strip()
                return Path(selected) if selected else None
            if result.returncode not in (1,):
                err = result.stderr.strip().splitlines()
                tail = err[-1] if err else "unknown kdialog error"
                print(f"[Export clips] kdialog failed: {tail}", flush=True)

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            print(f"[Export clips] Could not open directory picker: {e}", flush=True)
            return None

        root = tk.Tk()
        root.withdraw()
        try:
            # Respect display DPI so fallback dialogs are not tiny on HiDPI setups.
            try:
                dpi = float(root.winfo_fpixels("1i"))
                if dpi > 0:
                    root.tk.call("tk", "scaling", dpi / 72.0)
            except Exception:
                pass
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            selected = filedialog.askdirectory(
                title="Select directory for exported clips",
                initialdir=initial_dir,
                mustexist=True,
                parent=root,
            )
        except Exception as e:
            print(f"[Export clips] Directory picker failed: {e}", flush=True)
            selected = ""
        finally:
            root.destroy()

        if not selected:
            return None
        return Path(selected)

    def _clip_output_path(self, output_dir: Path, shot_idx: int) -> Path:
        """Return output filename using source#N.ext convention (1-based shot index)."""
        source_name = self.video_path.name
        source_path = Path(source_name)
        shot_number = shot_idx + 1
        if source_path.suffix:
            return output_dir / f"{source_path.stem}#{shot_number}{source_path.suffix}"
        return output_dir / f"{source_name}#{shot_number}"

    def _set_export_status(self, status: str) -> None:
        """Update short export status text shown in HUD."""
        with self._export_status_lock:
            self._export_status = status

    def _get_export_status(self) -> str:
        """Read current export status text."""
        with self._export_status_lock:
            return self._export_status

    def _is_export_running(self) -> bool:
        """Return True when an export thread is active."""
        return self._export_thread is not None and self._export_thread.is_alive()

    def _start_export_task(self, label: str, task: Callable[[], None]) -> bool:
        """Run a potentially long export task off the UI thread."""
        if self._is_export_running():
            print("[Export] Another export is already running.", flush=True)
            self._set_export_status("Export busy")
            return False

        def _runner() -> None:
            self._set_export_status(f"{label} in progress")
            try:
                task()
            except Exception as e:
                print(f"[{label}] Failed: {e}", flush=True)
                self._set_export_status(f"{label} failed")
                return
            # If worker did not publish a final state, mark completion.
            if self._get_export_status() == f"{label} in progress":
                self._set_export_status(f"{label} complete")

        self._export_thread = threading.Thread(target=_runner, daemon=True)
        self._export_thread.start()
        return True

    def _choose_export_file_path(self, default_filename: str) -> Optional[Path]:
        """Open a save-file picker and return selected output file path."""
        initial_dir = str(self.video_path.parent)
        initial_path = str(Path(initial_dir) / default_filename)

        zenity_path = shutil.which("zenity")
        if zenity_path is not None:
            cmd = [
                zenity_path,
                "--file-selection",
                "--save",
                "--confirm-overwrite",
                "--title=Save exported sequence as",
                f"--filename={initial_path}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                selected = result.stdout.strip()
                return Path(selected) if selected else None
            if result.returncode not in (1,):
                err = result.stderr.strip().splitlines()
                tail = err[-1] if err else "unknown zenity error"
                print(f"[Export sequence] zenity failed: {tail}", flush=True)

        kdialog_path = shutil.which("kdialog")
        if kdialog_path is not None:
            cmd = [
                kdialog_path,
                "--getsavefilename",
                initial_path,
                "--title",
                "Save exported sequence as",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                selected = result.stdout.strip()
                return Path(selected) if selected else None
            if result.returncode not in (1,):
                err = result.stderr.strip().splitlines()
                tail = err[-1] if err else "unknown kdialog error"
                print(f"[Export sequence] kdialog failed: {tail}", flush=True)

        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as e:
            print(f"[Export sequence] Could not open save-file picker: {e}", flush=True)
            return None

        root = tk.Tk()
        root.withdraw()
        try:
            try:
                dpi = float(root.winfo_fpixels("1i"))
                if dpi > 0:
                    root.tk.call("tk", "scaling", dpi / 72.0)
            except Exception:
                pass
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            selected = filedialog.asksaveasfilename(
                title="Save exported sequence as",
                initialdir=initial_dir,
                initialfile=default_filename,
                parent=root,
                defaultextension=self.video_path.suffix or ".mp4",
                filetypes=[
                    ("Video files", "*.mp4 *.mkv *.mov *.avi *.webm"),
                    ("All files", "*.*"),
                ],
            )
        except Exception as e:
            print(f"[Export sequence] Save-file picker failed: {e}", flush=True)
            selected = ""
        finally:
            root.destroy()

        if not selected:
            return None
        return Path(selected)

    def _export_clips(self, shot_indices: List[int]) -> None:
        """Launch background export for selected shots (no re-encode stream copy)."""
        if not shot_indices:
            return

        output_dir = self._choose_export_directory()
        if output_dir is None:
            if self.verbose:
                print("[Export clips] Cancelled.", flush=True)
            return

        ordered_unique = list(dict.fromkeys(shot_indices))
        print(
            f"[Export clips] Exporting {len(ordered_unique)} clip(s) to {output_dir}",
            flush=True,
        )

        self._start_export_task(
            "Export clips",
            lambda: self._export_clips_worker(ordered_unique, output_dir),
        )

    def _export_clips_worker(self, shot_indices: List[int], output_dir: Path) -> None:
        """Background worker: export selected shots via ffmpeg stream copy."""
        exported_count = 0
        failed_count = 0

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self._set_export_status("Export clips failed")
            print("[Export clips] ffmpeg not found in PATH.", flush=True)
            return

        try:
            for shot_idx in shot_indices:
                if shot_idx < 0 or shot_idx >= len(self.shot_timecodes):
                    continue

                # Export boundaries are always from scenedetect shot_timecodes.
                scene_start_time, scene_end_time = self.shot_timecodes[shot_idx]
                duration = scene_end_time - scene_start_time
                if duration <= 0:
                    continue

                output_path = self._clip_output_path(output_dir, shot_idx)
                try:
                    # Use stream copy to preserve source-encoded quality without re-encoding.
                    # Note: stream copy cuts can be keyframe-bound on some sources.
                    cmd = [
                        ffmpeg_path,
                        "-y",
                        "-i",
                        str(self.video_path),
                        "-ss",
                        f"{scene_start_time:.6f}",
                        "-t",
                        f"{duration:.6f}",
                        "-map",
                        "0:v?",
                        "-map",
                        "0:a?",
                        "-c",
                        "copy",
                        str(output_path),
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        err_lines = result.stderr.strip().splitlines()
                        detail = err_lines[-1] if err_lines else "unknown ffmpeg error"
                        raise RuntimeError(detail)

                    exported_count += 1
                    print(f"[Export clips] Shot {shot_idx + 1} -> {output_path.name}", flush=True)
                except Exception as e:
                    failed_count += 1
                    print(f"[Export clips] Shot {shot_idx + 1} failed: {e}", flush=True)
        except Exception as e:
            self._set_export_status("Export clips failed")
            print(f"[Export clips] Failed: {e}", flush=True)
            return

        if failed_count > 0:
            self._set_export_status(f"Export clips complete ({exported_count} ok, {failed_count} failed)")
        else:
            self._set_export_status(f"Export clips complete ({exported_count} clips)")

    def _export_clips_frame_accurate(self, shot_indices: List[int]) -> None:
        """Launch background export for selected shots with frame-accurate re-encode."""
        if not shot_indices:
            return

        output_dir = self._choose_export_directory()
        if output_dir is None:
            if self.verbose:
                print("[Export clips frame-accurate] Cancelled.", flush=True)
            return

        ordered_unique = list(dict.fromkeys(shot_indices))
        print(
            f"[Export clips frame-accurate] Exporting {len(ordered_unique)} clip(s) to {output_dir}",
            flush=True,
        )

        self._start_export_task(
            "Export clips frame-accurate",
            lambda: self._export_clips_frame_accurate_worker(ordered_unique, output_dir),
        )

    def _export_clips_frame_accurate_worker(self, shot_indices: List[int], output_dir: Path) -> None:
        """Background worker: export selected shots via MoviePy write_videofile."""
        exported_count = 0
        failed_count = 0

        try:
            for shot_idx in shot_indices:
                if shot_idx < 0 or shot_idx >= len(self.shot_timecodes):
                    continue

                scene_start_time, scene_end_time = self.shot_timecodes[shot_idx]
                if scene_end_time <= scene_start_time:
                    continue

                output_path = self._clip_output_path(output_dir, shot_idx)
                source_clip: Optional[VideoFileClip] = None
                shot_clip: Optional[VideoFileClip] = None
                try:
                    source_clip = VideoFileClip(str(self.video_path))
                    shot_clip = source_clip.subclipped(scene_start_time, scene_end_time)
                    if shot_clip is None:
                        raise RuntimeError("Failed to create subclip")
                    shot_clip.write_videofile(str(output_path), logger=None)
                    exported_count += 1
                    print(
                        f"[Export clips frame-accurate] Shot {shot_idx + 1} -> {output_path.name}",
                        flush=True,
                    )
                except Exception as e:
                    failed_count += 1
                    print(f"[Export clips frame-accurate] Shot {shot_idx + 1} failed: {e}", flush=True)
                finally:
                    if shot_clip is not None:
                        shot_clip.close()
                    if source_clip is not None:
                        source_clip.close()
        except Exception as e:
            self._set_export_status("Export clips frame-accurate failed")
            print(f"[Export clips frame-accurate] Failed: {e}", flush=True)
            return

        if failed_count > 0:
            self._set_export_status(
                f"Export clips frame-accurate complete ({exported_count} ok, {failed_count} failed)"
            )
        else:
            self._set_export_status(f"Export clips frame-accurate complete ({exported_count} clips)")

    def _export_sequence(self, shot_indices: List[int]) -> None:
        """Launch background export for selected shots as one sequence clip."""
        if not shot_indices:
            return

        ordered_unique = list(dict.fromkeys(shot_indices))
        source_path = Path(self.video_path.name)
        shot_numbers_suffix = "-".join(str(idx + 1) for idx in ordered_unique)
        default_name = (
            f"{source_path.stem}#sequence-{shot_numbers_suffix}{source_path.suffix}"
            if source_path.suffix
            else f"{source_path.name}#sequence-{shot_numbers_suffix}"
        )
        output_path = self._choose_export_file_path(default_name)
        if output_path is None:
            if self.verbose:
                print("[Export sequence] Cancelled.", flush=True)
            return

        print(
            f"[Export sequence] Exporting {len(ordered_unique)} clip(s) to {output_path}",
            flush=True,
        )

        self._start_export_task(
            "Export sequence",
            lambda: self._export_sequence_worker(ordered_unique, output_path),
        )

    def _export_sequence_worker(self, shot_indices: List[int], output_path: Path) -> None:
        """Background worker: compose and export one sequence clip."""

        source_clips: List[VideoFileClip] = []
        sequence_parts: List[VideoFileClip] = []
        sequence_clip = None
        try:
            for shot_idx in shot_indices:
                if shot_idx < 0 or shot_idx >= len(self.shot_timecodes):
                    continue

                scene_start_time, scene_end_time = self.shot_timecodes[shot_idx]
                if scene_end_time <= scene_start_time:
                    continue

                source_clip = VideoFileClip(str(self.video_path))
                part_clip = source_clip.subclipped(scene_start_time, scene_end_time)
                source_clips.append(source_clip)
                sequence_parts.append(part_clip)

            if not sequence_parts:
                print("[Export sequence] No valid clips to export.", flush=True)
                self._set_export_status("Export sequence failed")
                return

            sequence_clip = concatenate_videoclips(sequence_parts, method="compose")
            if sequence_clip is None:
                raise RuntimeError("Failed to compose sequence clip")
            sequence_clip.write_videofile(str(output_path), logger=None)
            print(f"[Export sequence] Done -> {output_path.name}", flush=True)
            self._set_export_status("Export sequence complete")
        except Exception as e:
            self._set_export_status("Export sequence failed")
            print(f"[Export sequence] Failed: {e}", flush=True)
        finally:
            if sequence_clip is not None:
                sequence_clip.close()
            for part in sequence_parts:
                part.close()
            for source in source_clips:
                source.close()

    def _open_sequence_viewer(self, shot_indices: List[int], start_position: int = 0) -> None:
        """Open a video-only viewer for the selected shots in sequence."""
        self._close_active_viewer()
        self._sequence_viewer = SequenceViewer(
            shot_indices=shot_indices,
            shot_timecodes=self.shot_timecodes,
            video_path=self.video_path,
            fps=self.fps,
            screen_w=self.window_width,
            screen_h=self.window_height,
            start_position=start_position,
        )
        selection_label = ", ".join(f"S{idx + 1}" for idx in shot_indices)
        if self.verbose:
            print(f"[SequenceViewer] Opening selected shots: {selection_label}", flush=True)

    def _clear_all_frame_caches(self) -> None:
        """Clear base + hi-res frame caches after a geometry change."""
        with self._cache_lock:
            self.frame_buffer.clear()
            self.frame_read_order.clear()
            self.latest_ready_frame.clear()
            self.latest_ready_idx.clear()
            self.hires_frame_buffer.clear()
            self.hires_frame_read_order.clear()
            self._visible_hires_requests = []
            self._visible_shot_indices = []
            self._visible_shot_weights = {}

    def _handle_window_resize(self, new_width: int, new_height: int) -> bool:
        """Reconfigure window-dependent layout and caches after a resize event.

        Returns True when window geometry actually changed.
        """
        if self.fullscreen:
            return False

        new_width = max(320, int(new_width))
        new_height = max(240, int(new_height))
        if new_width == self.window_width and new_height == self.window_height:
            return False

        old_cell_dims = self.cell_dims
        old_zoom = self.grid_zoom

        self.window_width = new_width
        self.window_height = new_height
        self.display = pygame.display.set_mode((new_width, new_height), self._display_flags)

        self.cell_dims, self.layout, self.pad_x, self.pad_y = calculate_grid_layout(
            len(self.shot_timecodes),
            self.window_width,
            self.window_height,
            self.video_width,
            self.video_height,
            allow_padding=self.allow_padding,
        )

        cell_w, cell_h = self.cell_dims
        self.grid_zoom_max = max(
            self.window_width / max(1, cell_w),
            self.window_height / max(1, cell_h),
        )
        self.grid_zoom = max(self.grid_zoom_min, min(self.grid_zoom_max, self.grid_zoom))
        self._clamp_grid_offset()

        self.bytes_per_cached_frame = cell_w * cell_h * 3
        self.estimated_full_cache_bytes = self.total_bucket_frames * self.bytes_per_cached_frame

        if old_cell_dims != self.cell_dims:
            self._clear_all_frame_caches()
            self._subtitle_overlay_cache.clear()
            self._close_active_viewer()

        self._maybe_prune_hires_cache_for_zoom(old_zoom, self.grid_zoom)
        if self.verbose:
            print(
                f"[Resize] {self.window_width}x{self.window_height} | Grid {self.layout[0]}x{self.layout[1]} | Cell {self.cell_dims[0]}x{self.cell_dims[1]}",
                flush=True,
            )
        return True

    def _close_active_viewer(self) -> None:
        """Close whichever overlay viewer is currently open."""
        if self._shot_viewer is not None:
            self._shot_viewer.close()
            self._shot_viewer = None
        if self._sequence_viewer is not None:
            self._sequence_viewer.close()
            self._sequence_viewer = None

    def _toggle_selected_shot(self, shot_idx: int) -> None:
        """Toggle a shot in the ordered selection list."""
        if shot_idx in self.selected_shot_indices:
            self.selected_shot_indices.remove(shot_idx)
        else:
            self.selected_shot_indices.append(shot_idx)

    def _hovered_shot_index(self, mouse_pos: Optional[Tuple[int, int]] = None) -> Optional[int]:
        """Return the 0-based shot index under the mouse cursor, if any."""
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        mouse_x, mouse_y = self._screen_to_grid_base(mouse_pos)
        cell_w, cell_h = self.cell_dims
        rows, cols = self.layout

        grid_w = cols * cell_w
        grid_h = rows * cell_h
        if mouse_x < self.pad_x or mouse_x >= self.pad_x + grid_w:
            return None
        if mouse_y < self.pad_y or mouse_y >= self.pad_y + grid_h:
            return None

        col = (mouse_x - self.pad_x) // cell_w
        row = (mouse_y - self.pad_y) // cell_h
        shot_idx = int(row * cols + col)
        if shot_idx < 0 or shot_idx >= len(self.shot_timecodes):
            return None
        return shot_idx

    def _screen_to_grid_base(self, mouse_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Map a screen-space point to unzoomed grid-space coordinates."""
        sx, sy = mouse_pos
        bx = (sx - self.grid_offset_x) / self.grid_zoom
        by = (sy - self.grid_offset_y) / self.grid_zoom
        return bx, by

    def _apply_grid_transform(self, x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
        """Apply current grid zoom+offset to a grid cell rectangle."""
        dx = int(x * self.grid_zoom + self.grid_offset_x)
        dy = int(y * self.grid_zoom + self.grid_offset_y)
        dw = max(1, int(w * self.grid_zoom))
        dh = max(1, int(h * self.grid_zoom))
        return dx, dy, dw, dh

    def _zoom_grid(self, factor: float, anchor: Optional[Tuple[int, int]] = None) -> None:
        """Zoom the grid in/out around an anchor point in screen coordinates."""
        if factor <= 0:
            return
        old_zoom = self.grid_zoom
        new_zoom = max(self.grid_zoom_min, min(self.grid_zoom_max, old_zoom * factor))
        if math.isclose(new_zoom, old_zoom):
            return

        if anchor is None:
            anchor = (self.window_width // 2, self.window_height // 2)
        ax, ay = anchor

        base_x = (ax - self.grid_offset_x) / old_zoom
        base_y = (ay - self.grid_offset_y) / old_zoom

        self.grid_zoom = new_zoom
        self.grid_offset_x = ax - base_x * new_zoom
        self.grid_offset_y = ay - base_y * new_zoom
        self._clamp_grid_offset()
        self._maybe_prune_hires_cache_for_zoom(old_zoom, new_zoom)

    def _snap_shot_to_window(self, shot_idx: int) -> None:
        """Toggle zoom between shot snap target and full-grid view (1.0)."""
        if shot_idx < 0 or shot_idx >= len(self.shot_timecodes):
            return

        old_zoom = self.grid_zoom

        rows, cols = self.layout
        cell_w, cell_h = self.cell_dims
        row = shot_idx // cols
        col = shot_idx % cols
        shot_center_x = self.pad_x + col * cell_w + cell_w / 2.0
        shot_center_y = self.pad_y + row * cell_h + cell_h / 2.0

        # Snap target is the maximum zoom that still shows the complete tile.
        snap_zoom = min(self.window_width / cell_w, self.window_height / cell_h) * 0.95
        snap_zoom = max(self.grid_zoom_min, min(self.grid_zoom_max, snap_zoom))
        snap_tolerance = max(0.01, snap_zoom * 0.03)

        if old_zoom < (snap_zoom - snap_tolerance):
            # Below snap target: zoom in and center clicked shot.
            self.grid_zoom = snap_zoom
            self.grid_offset_x = self.window_width / 2.0 - shot_center_x * snap_zoom
            self.grid_offset_y = self.window_height / 2.0 - shot_center_y * snap_zoom
        else:
            # At/near (or above) snap target: return to baseline full-grid zoom.
            self.grid_zoom = 1.0
            self.grid_offset_x = 0.0
            self.grid_offset_y = 0.0
        
        self._clamp_grid_offset()
        self._maybe_prune_hires_cache_for_zoom(old_zoom, self.grid_zoom)

    def _reset_grid_zoom(self) -> None:
        """Reset grid zoom and pan offset to default view."""
        old_zoom = self.grid_zoom
        self.grid_zoom = 1.0
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0
        self._maybe_prune_hires_cache_for_zoom(old_zoom, self.grid_zoom)

    def _grid_base_rect(self) -> pygame.Rect:
        """Return the unzoomed grid bounds in screen-space coordinates."""
        rows, cols = self.layout
        cell_w, cell_h = self.cell_dims
        return pygame.Rect(self.pad_x, self.pad_y, cols * cell_w, rows * cell_h)

    def _grid_transformed_size(self) -> Tuple[float, float]:
        """Return transformed grid width/height after zoom."""
        base = self._grid_base_rect()
        return base.width * self.grid_zoom, base.height * self.grid_zoom

    def _can_pan_grid(self) -> bool:
        """Whether transformed grid exceeds viewport and can be panned."""
        tw, th = self._grid_transformed_size()
        return tw > self.window_width or th > self.window_height

    def _clamp_grid_offset(self) -> None:
        """Clamp pan offsets so the grid stays within sensible viewport bounds."""
        base = self._grid_base_rect()
        scaled_w = base.width * self.grid_zoom
        scaled_h = base.height * self.grid_zoom

        overscroll_x = max(
            self.pan_overscroll_min_px,
            min(self.pan_overscroll_max_px, int(self.window_width * self.pan_overscroll_fraction)),
        )
        overscroll_y = max(
            self.pan_overscroll_min_px,
            min(self.pan_overscroll_max_px, int(self.window_height * self.pan_overscroll_fraction)),
        )

        if scaled_w <= self.window_width:
            self.grid_offset_x = 0.0
        else:
            min_offset_x = self.window_width - base.x - scaled_w - overscroll_x
            max_offset_x = -base.x + overscroll_x
            self.grid_offset_x = max(min_offset_x, min(max_offset_x, self.grid_offset_x))

        if scaled_h <= self.window_height:
            self.grid_offset_y = 0.0
        else:
            min_offset_y = self.window_height - base.y - scaled_h - overscroll_y
            max_offset_y = -base.y + overscroll_y
            self.grid_offset_y = max(min_offset_y, min(max_offset_y, self.grid_offset_y))

    def _pan_grid(self, dx: float, dy: float) -> None:
        """Pan transformed grid by screen-space delta and clamp to bounds."""
        self.grid_offset_x += dx
        self.grid_offset_y += dy
        self._clamp_grid_offset()

    def _update_keyboard_pan(self, dt: float) -> None:
        """Apply continuous panning while arrow keys are held."""
        if not self._can_pan_grid():
            return
        keys = pygame.key.get_pressed()
        speed = self.pan_pixels_per_second * dt
        dx = 0.0
        dy = 0.0
        if keys[pygame.K_LEFT]:
            dx += speed
        if keys[pygame.K_RIGHT]:
            dx -= speed
        if keys[pygame.K_UP]:
            dy += speed
        if keys[pygame.K_DOWN]:
            dy -= speed
        if dx != 0.0 or dy != 0.0:
            self._pan_grid(dx, dy)

    def _update_edge_pan(self, dt: float) -> None:
        """Auto-pan when mouse is near window edges."""
        if not self._can_pan_grid():
            return
        mx, my = pygame.mouse.get_pos()
        speed = self.pan_pixels_per_second * dt
        dx = 0.0
        dy = 0.0
        margin = self.edge_pan_margin
        if mx <= margin:
            dx += speed
        elif mx >= self.window_width - margin:
            dx -= speed
        if my <= margin:
            dy += speed
        elif my >= self.window_height - margin:
            dy -= speed
        if dx != 0.0 or dy != 0.0:
            self._pan_grid(dx, dy)

    def _hovered_shot_number(self) -> Optional[int]:
        """Return 1-based shot number under the mouse cursor, if any."""
        shot_idx = self._hovered_shot_index()
        if shot_idx is None:
            return None
        return shot_idx + 1
    
    def _render_hud(self):
        """Render heads-up display with playback info."""
        font = pygame.font.Font(None, 24)
        status = "PLAYING" if self.playing else "PAUSED"
        hovered_shot = self._hovered_shot_number()
        hover_text = f"S{hovered_shot}" if hovered_shot is not None else "-"
        selected_text = str(len(self.selected_shot_indices))
        zoom_text = f"{self.grid_zoom:.2f}x"
        export_status = self._get_export_status()
        export_suffix = f" | {export_status}" if export_status else ""
        text = font.render(
            f"{status} | {self.current_time:.1f}s / {self.max_duration:.1f}s | Hover: {hover_text} | Selected: {selected_text} | Zoom: {zoom_text} | Q=quit SPACE=pause BACKSPACE=clear{export_suffix}",
            True,
            (255, 255, 255)
        )
        self.display.blit(text, (10, 10))


def view_grid(
    video_path: Path,
    manifest_path: Optional[Path] = None,
    shot_timecodes: Optional[List[Tuple[float, float]]] = None,
    window_width: int = 1920,
    window_height: int = 1080,
    fps: Optional[int] = None,
    allow_padding: bool = True,
    fullscreen: bool = False,
    maximized: bool = False,
    verbose: bool = False,
    subtitle_path: Optional[Path] = None,
) -> None:
    """
    Open an interactive grid viewer for video shots from a manifest.
    
    Parameters
    ----------
    video_path : Path
        Path to the input video file
    manifest_path : Optional[Path]
        Path to the shotlist manifest (JSON/CSV). Required if shot_timecodes is not provided.
    shot_timecodes : Optional[List[Tuple[float, float]]]
        In-memory shot list as (start_time, end_time) seconds. If provided, manifest_path is ignored.
    window_width : int
        Window width in pixels
    window_height : int
        Window height in pixels
    fps : Optional[int]
        Frames per second for playback. If None, uses video's FPS
    allow_padding : bool
        Whether to preserve aspect ratio with padding
    fullscreen : bool
        Open viewer in fullscreen mode
    """
    if shot_timecodes is None:
        if manifest_path is None:
            raise ValueError("Either manifest_path or shot_timecodes must be provided")

        # Load manifest
        if verbose:
            print(f"Loading manifest: {manifest_path}")
        manifest = ShotListManifest(manifest_path)

        if not manifest.shots:
            raise ValueError(f"No shots found in manifest: {manifest_path}")

        # Get timecodes
        shot_timecodes = manifest.get_shot_timecodes()
        if verbose:
            print(f"Loaded {len(shot_timecodes)} shots from manifest")
    else:
        if not shot_timecodes:
            raise ValueError("No shots found in provided shot_timecodes")
        if verbose:
            print(f"Loaded {len(shot_timecodes)} shots from in-memory scene detection")
    
    # Create and run viewer
    viewer = GridViewer(
        video_path,
        shot_timecodes,
        window_width=window_width,
        window_height=window_height,
        fps=fps,
        allow_padding=allow_padding,
        fullscreen=fullscreen,
        maximized=maximized,
        verbose=verbose,
        subtitle_path=subtitle_path,
    )
    
    viewer.run()
