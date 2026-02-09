import math
import warnings
from typing import Optional

import typer

from pathlib import Path

from moviepy import VideoFileClip, CompositeVideoClip
from moviepy.video.fx.Loop import Loop
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import numpy as np

# Suppress MoviePy warnings about reading last frames of split videos
warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")

app = typer.Typer()


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
    video_file_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input video file",
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
        help="Target resolution as WIDTHxHEIGHT (e.g., 1920x1080). Defaults to input video size.",
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
    
    Detects scenes in a video, downscales them, and arranges them in an optimal grid layout.
    """
    print(f"Processing video: {video_file_path}")
    v = VideoFileClip(str(video_file_path))
    print(f"Video size: {v.size}")

    # Capture input video parameters
    input_fps = v.fps
    print(f"Input fps: {input_fps}")

    # Parse resolution or use video size
    if resolution:
        try:
            res_parts = resolution.lower().split('x')
            if len(res_parts) != 2:
                raise ValueError("Resolution must be in format WIDTHxHEIGHT")
            target_resolution = (int(res_parts[0]), int(res_parts[1]))
        except (ValueError, IndexError) as e:
            typer.echo(f"Error: Invalid resolution format '{resolution}'. Use WIDTHxHEIGHT (e.g., 1920x1080)", err=True)
            raise typer.Exit(1)
    else:
        video_w, video_h = int(v.size[0]), int(v.size[1])
        target_resolution: tuple[int, int] = (video_w, video_h)

    print(f"Target resolution: {target_resolution[0]}x{target_resolution[1]}")

    # Detect scenes
    scene_list = detect(str(video_file_path), AdaptiveDetector())
    print(f"Detected {len(scene_list)} scenes")

    if not scene_list:
        typer.echo("Error: No scenes detected in video", err=True)
        raise typer.Exit(1)

    # Calculate optimal grid layout and clip dimensions
    (target_dims, layout, pad_x, pad_y) = small_multiples(
        count=len(scene_list),
        resolution=(2048, 1556),
        size=target_resolution,
        allow_padding=padding
    )
    rows, cols = layout
    print(f"Target clip dimensions: {target_dims}")
    print(f"Grid layout: {rows} rows x {cols} cols")

    # Split video into scene clips
    output_dir = video_file_path.parent / 'scenes'
    output_dir.mkdir(exist_ok=True)
    split_video_ffmpeg(str(video_file_path), scene_list, output_file_template=str(output_dir / '$SCENE_NUMBER.mp4'))

    # Load and downscale each scene clip
    downscaled_clips = []
    for i in range(len(scene_list)):
        scene_file = output_dir / f'{i+1:03d}.mp4'
        if scene_file.exists():
            clip = VideoFileClip(str(scene_file))
            # Resize to target dimensions
            clip = clip.resized(target_dims)
            downscaled_clips.append(clip)
        else:
            print(f"Warning: Scene file {scene_file} not found")

    print(f"Loaded and downscaled {len(downscaled_clips)} clips")

    # Optionally loop clips to match the longest clip duration
    if loop_clips and downscaled_clips:
        max_duration = max(clip.duration for clip in downscaled_clips)
        print(f"Longest clip duration: {max_duration:.2f}s")

        looped_clips = []
        for clip in downscaled_clips:
            if clip.duration < max_duration:
                # Loop the clip to match the longest duration
                clip = clip.with_effects([Loop(duration=max_duration)])
            looped_clips.append(clip)
        downscaled_clips = looped_clips

    # Create grid layout with packed rectangles
    target_w, target_h = target_dims
    canvas_w, canvas_h = target_resolution

    positioned_clips = []
    for idx, clip in enumerate(downscaled_clips):
        row = idx // cols
        col = idx % cols

        # Calculate position in grid with padding
        x = pad_x + col * target_w
        y = pad_y + row * target_h

        # Position the clip
        positioned_clip = clip.with_position((x, y))
        positioned_clips.append(positioned_clip)

    # Composite all clips into a single video
    final_clip = CompositeVideoClip(positioned_clips, size=(canvas_w, canvas_h))

    # Write output using input video parameters
    print(f"Writing output to: {output}")
    final_clip.write_videofile(str(output), codec='libx264', fps=input_fps, audio=False)

    # Cleanup
    v.close()
    for clip in downscaled_clips:
        clip.close()
    final_clip.close()

    # Remove interim scene clips if requested
    if cleanup:
        import shutil
        print(f"Cleaning up scene files in {output_dir}...")
        shutil.rmtree(output_dir)
        print("Scene files removed.")
    
    print("Done!")


if __name__ == "__main__":
    app()
