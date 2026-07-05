# Reel Curator

Local macOS travel video curation for finding the best Instagram Reel candidates in a large iPhone video folder. The app is designed to run entirely on your Mac: videos are scanned locally, thumbnails and analysis results are cached locally, and optional Apple Vision analysis uses on-device macOS frameworks.

## What It Does

- Scans `.mov`, `.mp4`, `.m4v`, `.avi`, `.mkv`, and common video formats.
- Computes duration, resolution, frame rate, file size, brightness, sharpness, stability, motion, scene changes, orientation, and thumbnails.
- Corrects iPhone portrait/landscape orientation using ffprobe rotation metadata.
- Can skip landscape videos during scanning when you only want portrait Reel candidates.
- Creates perceptual fingerprints for near-duplicate grouping.
- Scores each video from 0-100, with people/faces as the highest-priority signal.
- Lets you add local reference face photos for people whose clips should receive stronger scores.
- Provides a Streamlit visual review UI with filters, sorting, favorites, rejects, comparison, playback, and exports.
- Exports favorites, top N, best duplicate representatives, CSV, JSON, and contact sheet PDF.
- Caches work so interrupted or repeated scans continue quickly.

## Local Setup

Use Python 3.12+.

```bash
brew install ffmpeg python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[apple-vision,dev]"
```

The Apple Vision extra is optional. Without it, the app still performs OpenCV-based analysis and heuristic tagging.

## Run The App

```bash
reel-curator ui
```

Or:

```bash
streamlit run src/reel_curator/ui/app.py
```

Default input folder:

```text
/Users/kunjain/Movies/Alberta videos
```

Change it in the sidebar or in `config.yaml`.

## CLI Scan

```bash
reel-curator scan --input "/Users/kunjain/Movies/Alberta videos"
```

## Architecture

- `scanner.py`: Finds video files and tracks filesystem identity.
- `video_probe.py`: Reads technical metadata with `ffprobe` when available and OpenCV fallback.
- `frame_sampler.py`: Samples frames and chooses a strong thumbnail frame.
- `metrics.py`: Brightness, blur/sharpness, motion, stability, scene changes, and perceptual hashes.
- `analyzers/vision.py`: Optional Apple Vision framework adapter for local on-device labels/observations.
- `tagging.py`: Combines Vision results and visual heuristics into tags.
- `scoring.py`: Produces transparent 0-100 scoring.
- `face_priority.py`: Local OpenCV face detection and reference-face matching.
- `duplicates.py`: Groups near-duplicates and recommends the best representative.
- `cache.py`: SQLite-backed persistent analysis cache.
- `exports.py`: Copies selected videos and writes CSV, JSON, and contact sheet PDF.
- `ui/app.py`: Streamlit review application.

## Score Formula

The default score is:

- People priority: 30%
- Image quality: 20%
- Stability: 14%
- Lighting: 10%
- Scenic content: 10%
- Uniqueness: 7%
- Composition/orientation usefulness: 4%
- Travel Reel usefulness: 5%

Each component is stored in the cache and shown in exports so the ranking is explainable.

If old iPhone portrait clips show as landscape, click **Force Rescan** after updating. The
previous orientation labels were cached before rotation metadata was applied.

## Privacy

No network calls are made by the application. Optional AI models should be installed locally. If you later add CLIP, SigLIP, BLIP, or Florence-2, keep model downloads as an explicit setup step and run inference from local weights.
