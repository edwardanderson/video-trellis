# Video Trellis

Draw the scenes of a video as an animated grid of small multiples.

![Colour in Clay (1950)](colour-in-clay-1950.gif)

G. B. Instructional. (1950). Colour in Clay. British Council. https://film.britishcouncil.org/resources/film-archive

> _Colour in Clay_ is part of the British Council film archive of short documentaries made by the British Council during the 1940s. The films were designed to show the world how Britain lived, worked and played. View, download and play with the archive at <https://film.britishcouncil.org/resources/film-archive>.

## CLI

### Encode

Create a new video file with scenes arranged in a grid:

```bash
video-trellis encode --input path/to/file --loop-clips --resolution 2048x1080 --output path/to/outfile
```

### View

View all scenes in an interactive grid without encoding:

```bash
video-trellis view --input /path/to/video.mp4
```

Optionally persist the `pyscenedetect` manifest:

```bash
uv run scenedetect -i /path/to/video.mp4 detect-adaptive list-scenes -f shots.csv
# Use `--manifest` to pass the shotlist
video-trellis view --input /path/to/video.mp4 --manifest shots.csv
```

Optionally render subtitles:

```bash
video-trellis view --input /path/to/video.mp4 --srt /path/to/subtitles.srt
```

#### UI

- double click to snap to a clip
- single click to select a clip
- right click for preview and export options for the selected clip(s)
- toggle subtitles on/off with `s`
- backspace deselects all selected clips
- `0` snaps to max zoom out
- zoom in/out with `+` / `-` or mouse
- `ESC` or `q` to exit
