"""
Real-time grid viewer for video shots using Pygame.

This module provides dynamic, GPU-accelerated viewing of video shots
arranged in a grid without requiring re-encoding.
"""

import csv
import json
import math
import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque

import pygame
import numpy as np
from moviepy import VideoFileClip
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


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
        
        # Load video
        self.video = VideoFileClip(str(video_path), audio=False)
        self.video_width = int(self.video.w)
        self.video_height = int(self.video.h)
        self.fps = fps or self.video.fps
        
        # Pygame setup
        pygame.init()
        if self.fullscreen:
            self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.window_width, self.window_height = self.display.get_size()
        else:
            self.display = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Video Grid Viewer - {video_path.name}")
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
        
        # Warm up caches with first frame of each shot
        print("Warming up frame caches...")
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
        self._warmup_frame_caches()
        print("✓ Frame caches ready")

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
        start_time, end_time = self.shot_timecodes[shot_idx]
        shot_duration = end_time - start_time
        if shot_duration <= 0:
            return start_time
        time_offset = min(frame_idx * self.cache_time_quantum, max(shot_duration - 0.001, 0.0))
        return start_time + time_offset

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
        frame = self._get_or_decode_frame(shot_idx, frame_idx, frame_time, source_video=source_video)
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
        
        This amortizes the cost of expensive MoviePy seeks during initialization
        so playback hits the cache instead of seeking every frame.
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
            
            # Progress indicator
            if (shot_idx + 1) % max(1, len(self.shot_timecodes) // 10) == 0:
                print(f"  Warmed {shot_idx + 1}/{len(self.shot_timecodes)} shots (~{decoded_count}/{total_to_decode} frames)")
    
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
                shot_count = len(self.shot_timecodes)
                if shot_count == 0:
                    time.sleep(0.01)
                    continue

                order = [
                    (self.prefetch_cursor + i) % shot_count
                    for i in range(shot_count)
                ]
                decoded_this_cycle = 0
                # Keep budget bounded to avoid one long cycle starving tail shots.
                cycle_budget = max(4, min(self.prefetch_cycle_decode_budget, shot_count))
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
                self.prefetch_cursor = (self.prefetch_cursor + cycle_budget) % shot_count
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.005)
            except Exception:
                # Silently ignore errors in background thread
                pass
    
    def get_shot_frame(self, shot_idx: int, time_offset: float, frame_count: int) -> Optional[np.ndarray]:
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
            
            # Resize to cell dimensions
            cell_w, cell_h = self.cell_dims
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
            
            # Culling: Skip cells that are completely off-screen
            if x + cell_w < 0 or x >= self.window_width or y + cell_h < 0 or y >= self.window_height:
                continue
            
            # Calculate playback position within this shot (looping)
            # Each shot loops independently as global time advances
            time_in_shot = self.current_time % shot_duration
            
            # Get frame (will use cache if available)
            frame = self.get_shot_frame(shot_idx, time_in_shot, frame_count)
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
                # Blit to surface
                surface.blit(frame_surface, (x, y))
            except Exception as e:
                # Silently skip if blit fails
                pass
        
        render_time = (time.perf_counter() - render_start) * 1000  # Convert to ms
        self.render_times.append(render_time)
        return surface
    
    def run(self):
        """Run the interactive grid viewer."""
        print(f"Grid Viewer: {len(self.shot_timecodes)} shots, {self.layout[0]}×{self.layout[1]} grid")
        print(f"Cell size: {self.cell_dims[0]}×{self.cell_dims[1]}")
        print(f"Target FPS: {self.fps}")
        print(f"Controls: SPACE=pause/play, LEFT/RIGHT=seek, Q=quit")
        
        # Start background prefetching thread
        self._start_prefetch_thread()
        
        running = True
        frame_count = 0
        loop_start_time = time.perf_counter()
        
        while running:
            frame_loop_start = time.perf_counter()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_LEFT:
                        self.current_time = max(0, self.current_time - 1.0)
                    elif event.key == pygame.K_RIGHT:
                        self.current_time = min(self.max_duration, self.current_time + 1.0)
            
            # Update playback time
            if self.playing:
                self.current_time += 1.0 / self.fps
                if self.current_time >= self.max_duration:
                    self.current_time = 0.0  # Loop
            
            # Render and display
            frame_surface = self.render_frame(frame_count)
            self.display.blit(frame_surface, (0, 0))
            
            # Display HUD
            self._render_hud()
            
            pygame.display.flip()
            self.clock.tick(self.fps)
            
            frame_count += 1
            
            # Track frame time
            frame_time = (time.perf_counter() - frame_loop_start) * 1000  # Convert to ms
            self.frame_times.append(frame_time)
            
            # Log performance metrics periodically (every N frames or on first frame)
            if frame_count == 1 or frame_count % self.perf_log_interval == 0:
                self._log_performance(frame_count)
        
        # Stop background thread and cleanup
        self._stop_prefetch_thread()
        pygame.quit()
        self.video.close()
        
        # Final performance summary
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

    def _hovered_shot_number(self) -> Optional[int]:
        """Return 1-based shot number under the mouse cursor, if any."""
        mouse_x, mouse_y = pygame.mouse.get_pos()
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
        return shot_idx + 1
    
    def _render_hud(self):
        """Render heads-up display with playback info."""
        font = pygame.font.Font(None, 24)
        status = "PLAYING" if self.playing else "PAUSED"
        hovered_shot = self._hovered_shot_number()
        hover_text = f"S{hovered_shot}" if hovered_shot is not None else "-"
        text = font.render(
            f"{status} | {self.current_time:.1f}s / {self.max_duration:.1f}s | Hover: {hover_text} | Q=quit SPACE=pause",
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
        print(f"Loading manifest: {manifest_path}")
        manifest = ShotListManifest(manifest_path)

        if not manifest.shots:
            raise ValueError(f"No shots found in manifest: {manifest_path}")

        # Get timecodes
        shot_timecodes = manifest.get_shot_timecodes()
        print(f"Loaded {len(shot_timecodes)} shots from manifest")
    else:
        if not shot_timecodes:
            raise ValueError("No shots found in provided shot_timecodes")
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
    )
    
    viewer.run()
