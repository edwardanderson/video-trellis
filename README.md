# Video Trellis

Draw the scenes of a video as an animated grid of small multiples.

![Colour in Clay (1950)](colour-in-clay-1950.gif)

G. B. Instructional. (1950). Colour in Clay. British Council. https://film.britishcouncil.org/resources/film-archive

> _Colour in Clay_ is part of the British Council film archive of short documentaries made by the British Council during the 1940s. The films were designed to show the world how Britain lived, worked and played. View, download and play with the archive at <https://film.britishcouncil.org/resources/film-archive>.

## CLI

### Encode Grid Video (Original Mode)

Create an encoded video file with scenes arranged in a grid:

```bash
video-trellis --input path/to/file --loop-clips --resolution 2048x1080 --output path/to/outfile
```

### View Grid Interactively (No Re-encoding)

View all scenes in an interactive grid without encoding:

```bash
# First, generate a scene manifest with pyscenedetect
uv run scenedetect -i video.mp4 detect-adaptive list-scenes -f shots.csv

# Then, view in the grid viewer
uv run python3 -m video_trellis.cli view-grid-cmd --input video.mp4 --manifest shots.csv
```

The grid viewer:
- **No re-encoding**: Dynamically reads and resizes frames in real-time
- **Fast startup**: Starts playing immediately without encoding delay
- **Interactive**: Play/pause, seek, and navigate through scenes
- **Smart layout**: Automatically calculates optimal grid dimensions

See [GRID_VIEWER.md](GRID_VIEWER.md) for detailed grid viewer documentation.
