import math
import subprocess
import warnings
from typing import Optional

import typer

from pathlib import Path

from moviepy import VideoFileClip
from scenedetect import detect, AdaptiveDetector
import numpy as np

from video_trellis.grid_viewer import view_grid

# Suppress MoviePy warnings about reading last frames of split videos
warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")

app = typer.Typer()


def split_and_scale_scenes(
    video_path: Path,
    scene_list: list,
    output_dir: Path,
    target_size: tuple[int, int],
    show_progress: bool = False
) -> int:
    """
    Split video into scenes and scale them using FFmpeg directly.
    Returns number of successfully created clips.
    """
    output_dir.mkdir(exist_ok=True)
    target_w, target_h = target_size
    clips_created = 0
    
    for idx, (start_time, end_time) in enumerate(scene_list, start=1):
        output_file = output_dir / f"{idx:03d}.mp4"
        
        # Convert FrameTimecode to seconds
        start_sec = start_time.get_seconds()
        end_sec = end_time.get_seconds()
        duration = end_sec - start_sec
        
        if duration <= 0:
            continue
        
        # Build FFmpeg command with robust parameters for MoviePy compatibility
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # Optimize for streaming/reading
            "-an",  # No audio (faster)
            str(output_file)
        ]
        
        if not show_progress:
            cmd.insert(1, "-loglevel")
            cmd.insert(2, "error")
        
        try:
            subprocess.run(cmd, check=True, capture_output=not show_progress)
            clips_created += 1
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to create scene {idx}: {e}")
            continue
    
    return clips_created


def probe_video_codec(video_path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    codec_name = result.stdout.strip()
    return codec_name if codec_name else None


def resolve_output_codec(codec_option: str, video_file_paths: list[Path]) -> str:
    if codec_option != "auto":
        return codec_option

    codec_map = {
        "h264": "libx264",
        "hevc": "libx265",
        "mpeg4": "mpeg4",
        "vp9": "libvpx-vp9",
        "av1": "libaom-av1",
    }

    input_codecs = {probe_video_codec(video_path) for video_path in video_file_paths}
    input_codecs.discard(None)

    if len(input_codecs) == 1:
        input_codec = next(iter(input_codecs))
        mapped_codec = codec_map.get(input_codec)
        if mapped_codec:
            print(f"All inputs share codec '{input_codec}', using output codec '{mapped_codec}'")
            return mapped_codec

    print("Input codecs differ or could not be probed, falling back to 'libx264'")
    return "libx264"


def small_multiples(
    count: int,
    resolution: tuple[int, int],
    size: tuple[int, int],
    allow_padding: bool = True,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    """
    Determine optimal grid layout and largest clip scale that fits all clips.

    Returns
    -------
    ((scaled_w, scaled_h), (rows, cols), pad_x, pad_y)
    """

    video_w, video_h = resolution
    canvas_w, canvas_h = size
    video_aspect = video_w / video_h

    best_scale = 0.0
    best_dims = (0, 0)
    best_layout = (0, 0)

    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)

        cell_w = canvas_w / cols
        cell_h = canvas_h / rows

        if allow_padding:
            # Preserve aspect ratio inside cell
            scale = min(cell_w / video_w, cell_h / video_h)

            scaled_w = video_w * scale
            scaled_h = video_h * scale

        else:
            # Require exact aspect match
            cell_aspect = cell_w / cell_h
            if not math.isclose(cell_aspect, video_aspect, rel_tol=1e-6):
                continue

            scale = cell_w / video_w
            scaled_w = cell_w
            scaled_h = cell_h

        if scale > best_scale:
            best_scale = scale
            best_dims = (int(scaled_w), int(scaled_h))
            best_layout = (rows, cols)

    # Calculate balanced padding
    rows, cols = best_layout
    target_w, target_h = best_dims
    grid_w = cols * target_w
    grid_h = rows * target_h

    if allow_padding:
        pad_x = (canvas_w - grid_w) // 2
        pad_y = (canvas_h - grid_h) // 2
    else:
        pad_x = 0
        pad_y = 0

    return best_dims, best_layout, pad_x, pad_y


@app.command()
def main(
    video_file_paths: list[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path(s) to input video file(s). Can be specified multiple times.",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    padding: bool = typer.Option(
        True,
        "--padding/--no-padding",
        "-p/-np",
        help="Allow padding to preserve aspect ratio in grid cells",
    ),
    loop_clips: bool = typer.Option(
        False,
        "--loop-clips/--no-loop",
        "-l/-nl",
        help="Loop shorter clips to match the longest clip duration",
    ),
    resolution: Optional[str] = typer.Option(
        None,
        "--resolution",
        "-r",
        help="Target resolution as WIDTHxHEIGHT (e.g., 1920x1080). Defaults to first input video size.",
    ),
    quality: int = typer.Option(
        23,
        "--quality",
        "-q",
        help="Video quality (0-51). Lower is better quality, slower encoding. 23=default, 28=faster/lower quality, 18=slower/high quality.",
        min=0,
        max=51,
    ),
    speed: str = typer.Option(
        "fast",
        "--speed",
        "-s",
        help="Encoding speed preset: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow.",
    ),
    codec: str = typer.Option(
        "libx264",
        "--codec",
        help="Output codec. Use 'auto' to map from shared input codec (e.g. h264 -> libx264).",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup/--no-cleanup",
        "-c/-nc",
        help="Remove interim scene clips after processing",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to output video file",
    ),
):
    """
    Create a trellis chart visualisation from video scenes.
    
    Detects scenes in one or more videos, downscales them, and arranges them in an optimal grid layout.
    """
    if not video_file_paths:
        typer.echo("Error: At least one input video is required", err=True)
        raise typer.Exit(1)

    selected_codec = resolve_output_codec(codec, video_file_paths)

    print(f"Processing {len(video_file_paths)} video(s)...")
    
    # PHASE 1: Detect scenes in all videos to determine total count
    print("\n=== Phase 1: Scene Detection ===")
    all_videos = []
    video_metadata = []  # List of (video_idx, video_path, scene_list, output_dir)
    video_fps = None
    base_resolution = None
    video_resolution = None

    for video_idx, video_file_path in enumerate(video_file_paths):
        print(f"\n[Video {video_idx + 1}/{len(video_file_paths)}] Detecting scenes: {video_file_path}")
        v = VideoFileClip(str(video_file_path))
        all_videos.append(v)
        print(f"  Video size: {v.size}")

        # Capture metadata from first video
        if video_fps is None:
            video_fps = v.fps
            video_resolution = v.size
            print(f"  FPS: {video_fps}")

        # Set base resolution from first video if not specified
        if base_resolution is None:
            if resolution:
                try:
                    res_parts = resolution.lower().split('x')
                    if len(res_parts) != 2:
                        raise ValueError("Resolution must be in format WIDTHxHEIGHT")
                    base_resolution = (int(res_parts[0]), int(res_parts[1]))
                except (ValueError, IndexError) as e:
                    typer.echo(f"Error: Invalid resolution format '{resolution}'. Use WIDTHxHEIGHT (e.g., 1920x1080)", err=True)
                    raise typer.Exit(1)
            else:
                video_w, video_h = int(v.size[0]), int(v.size[1])
                base_resolution = (video_w, video_h)
            print(f"  Target resolution: {base_resolution[0]}x{base_resolution[1]}")

        # Detect scenes
        scene_list = detect(str(video_file_path), AdaptiveDetector(), show_progress=True)
        print(f"  Detected {len(scene_list)} scenes")

        if not scene_list:
            typer.echo(f"  Warning: No scenes detected in video {video_idx + 1}", err=True)
            continue

        output_dir = video_file_path.parent / f'scenes_video_{video_idx}'
        video_metadata.append((video_idx, video_file_path, scene_list, output_dir))

    if not video_metadata:
        typer.echo("Error: No scenes detected in any video", err=True)
        raise typer.Exit(1)

    total_scenes = sum(len(scene_list) for _, _, scene_list, _ in video_metadata)
    print(f"\nTotal scenes across all videos: {total_scenes}")

    # Calculate optimal grid layout and clip dimensions
    (target_dims, layout, pad_x, pad_y) = small_multiples(
        count=total_scenes,
        resolution=video_resolution,
        size=base_resolution,
        allow_padding=padding
    )
    
    # Ensure dimensions are even (required for h264 encoding)
    target_w, target_h = target_dims
    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2
    target_dims = (target_w, target_h)
    
    rows, cols = layout
    print(f"Optimal grid layout: {rows} rows × {cols} cols")
    print(f"Target clip dimensions: {target_dims[0]}×{target_dims[1]} (rounded to even for h264)")

    # PHASE 2: Split videos with pre-scaling
    print("\n=== Phase 2: Splitting & Scaling Scenes ===")
    all_scene_files = []  # List of (scene_file, video_index, output_dir) tuples
    
    for video_idx, video_file_path, scene_list, output_dir in video_metadata:
        print(f"\n[Video {video_idx + 1}/{len(video_metadata)}] Splitting and scaling: {video_file_path.name}")
        print(f"  Creating {len(scene_list)} scene clips at {target_dims[0]}×{target_dims[1]}...")
        
        clips_created = split_and_scale_scenes(
            video_file_path,
            scene_list,
            output_dir,
            target_dims,
            show_progress=False
        )
        
        print(f"  Successfully created {clips_created}/{len(scene_list)} clips")

        # Track scene files for this video
        for i in range(len(scene_list)):
            scene_file = output_dir / f'{i+1:03d}.mp4'
            if scene_file.exists():
                all_scene_files.append((scene_file, video_idx, output_dir))

    print(f"\nTotal scene files created: {len(all_scene_files)}")

    # PHASE 3: Load pre-scaled scene clips
    print(f"\n=== Phase 3: Loading Pre-Scaled Clips ===")
    print(f"Loading {len(all_scene_files)} scene clips (already scaled to {target_dims[0]}×{target_dims[1]})...")
    downscaled_clips = []
    failed_clips = []
    
    for scene_file, video_idx, _ in all_scene_files:
        if scene_file.exists():
            try:
                clip = VideoFileClip(str(scene_file))
                # Clips are already scaled during split - no .resized() needed!
                downscaled_clips.append(clip)
            except Exception as e:
                print(f"  Warning: Failed to load {scene_file.name}: {e}")
                failed_clips.append(scene_file)
        else:
            print(f"  Warning: Scene file {scene_file} not found")
            failed_clips.append(scene_file)

    if failed_clips:
        print(f"\nWarning: {len(failed_clips)} clips failed to load and will be skipped")
    
    print(f"Successfully loaded {len(downscaled_clips)} clips")

    # PHASE 4: Frame-by-frame composite and render
    print(f"\n=== Phase 4: Frame-by-Frame Rendering ===")
    
    # Calculate clip durations and positions
    target_w, target_h = target_dims
    canvas_w, canvas_h = base_resolution
    
    clip_info = []  # List of (clip, x, y, duration, frame_count)
    max_duration = 0
    
    for idx, clip in enumerate(downscaled_clips):
        row = idx // cols
        col = idx % cols
        x = pad_x + col * target_w
        y = pad_y + row * target_h
        
        duration = clip.duration
        frame_count = int(duration * video_fps)
        
        if loop_clips:
            max_duration = max(max_duration, duration)
        
        clip_info.append((clip, x, y, duration, frame_count))
    
    # Determine total frames to render
    if loop_clips and max_duration > 0:
        total_frames = int(max_duration * video_fps)
        print(f"Rendering {total_frames} frames ({max_duration:.2f}s) with looped clips")
    else:
        # Use shortest clip duration if not looping
        total_frames = min(info[4] for info in clip_info) if clip_info else 0
        print(f"Rendering {total_frames} frames ({total_frames/video_fps:.2f}s) without looping")
    
    if total_frames == 0:
        typer.echo("Error: No frames to render", err=True)
        raise typer.Exit(1)
    
    # Setup FFmpeg process for encoding
    print(f"Encoding with codec '{selected_codec}', preset '{speed}', quality {quality}...")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{canvas_w}x{canvas_h}",
        "-pix_fmt", "rgb24",
        "-r", str(video_fps),
        "-i", "-",  # stdin
        "-an",  # no audio
        "-c:v", selected_codec,
        "-preset", speed,
        "-crf", str(quality),
        "-pix_fmt", "yuv420p",
        str(output)
    ]
    
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Render frames
    print(f"Rendering {len(downscaled_clips)} clips into {rows}×{cols} grid...")
    
    try:
        for frame_idx in range(total_frames):
            # Create blank canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Composite each clip onto canvas
            for clip, x, y, duration, frame_count in clip_info:
                # Calculate which frame to extract from this clip
                if loop_clips:
                    # Loop the clip
                    clip_time = (frame_idx / video_fps) % duration
                else:
                    clip_time = frame_idx / video_fps
                    if clip_time >= duration:
                        continue  # Clip has ended
                
                try:
                    # Get frame from clip
                    frame = clip.get_frame(clip_time)
                    
                    # Ensure frame is correct size (should already be from pre-scaling)
                    if frame.shape[:2] != (target_h, target_w):
                        # Fallback resize if needed
                        from PIL import Image
                        img = Image.fromarray(frame)
                        img = img.resize((target_w, target_h))
                        frame = np.array(img)
                    
                    # Blit frame onto canvas
                    canvas[y:y+target_h, x:x+target_w] = frame
                    
                except Exception as e:
                    # Clip failed, skip (already handled in loading phase)
                    pass
            
            # Write frame to FFmpeg
            ffmpeg_process.stdin.write(canvas.tobytes())
            
            # Progress indicator
            if (frame_idx + 1) % max(1, total_frames // 20) == 0:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx + 1}/{total_frames} frames)")
        
        # Close FFmpeg stdin to signal completion
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        if ffmpeg_process.returncode != 0:
            stderr = ffmpeg_process.stderr.read().decode()
            print(f"FFmpeg error: {stderr}")
            typer.echo("Error: FFmpeg encoding failed", err=True)
            raise typer.Exit(1)
        
        print(f"Successfully rendered {total_frames} frames to {output}")
        
    except Exception as e:
        ffmpeg_process.kill()
        raise

    # Cleanup
    for v in all_videos:
        v.close()
    for clip in downscaled_clips:
        clip.close()

    # Remove interim scene clips if requested
    if cleanup:
        import shutil
        scene_dirs = set(scene_dir for _, _, scene_dir in all_scene_files)
        for output_dir in scene_dirs:
            print(f"Cleaning up scene files in {output_dir}...")
            shutil.rmtree(output_dir)
        print("Scene files removed.")
    
    print("\nDone!")


@app.command()
def view_grid_cmd(
    video_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input video file",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        "-m",
        help="Optional path to pyscenedetect shotlist manifest (JSON/CSV). If omitted, scenes are detected on the fly.",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    width: int = typer.Option(
        1920,
        "--width",
        "-w",
        help="Window width in pixels",
        min=640,
    ),
    height: int = typer.Option(
        1080,
        "--height",
        "-h",
        help="Window height in pixels",
        min=480,
    ),
    fps: Optional[int] = typer.Option(
        None,
        "--fps",
        "-f",
        help="Playback FPS (defaults to video FPS)",
        min=1,
    ),
    fullscreen: bool = typer.Option(
        False,
        "--fullscreen/--windowed",
        help="Open the grid viewer in fullscreen mode",
    ),
    padding: bool = typer.Option(
        True,
        "--padding/--no-padding",
        "-p/-np",
        help="Preserve aspect ratio with padding",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging for cache, warmup, and performance diagnostics",
    ),
):
    """
    View video shots in an interactive grid without re-encoding.
    
    Displays shots dynamically in a grid layout using Pygame for real-time playback.
    If --manifest is omitted, scenes are detected in memory via pyscenedetect and
    passed directly to the viewer (no manifest file is written to disk).
    
    Controls:
    - SPACE: Play/pause
    - LEFT/RIGHT: Seek backward/forward by 1 second
    - Q: Quit
    """
    try:
        if manifest is not None:
            view_grid(
                video_file,
                manifest_path=manifest,
                window_width=width,
                window_height=height,
                fps=fps,
                allow_padding=padding,
                fullscreen=fullscreen,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"No manifest provided; detecting scenes in-memory: {video_file}")
            scene_list = detect(str(video_file), AdaptiveDetector(), show_progress=True)
            if not scene_list:
                raise ValueError("No scenes detected in input video")

            shot_timecodes = [
                (start_time.get_seconds(), end_time.get_seconds())
                for start_time, end_time in scene_list
            ]
            if verbose:
                print(f"Detected {len(shot_timecodes)} shots from input video")

            view_grid(
                video_file,
                shot_timecodes=shot_timecodes,
                window_width=width,
                window_height=height,
                fps=fps,
                allow_padding=padding,
                fullscreen=fullscreen,
                verbose=verbose,
            )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
