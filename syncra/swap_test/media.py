"""Render the four input versions (O, A, V, AV) of every swap-test case with ffmpeg.

    python -m syncra.swap_test.media --output-dir media \\
        --daily-omni /path/to/Daily-Omni --lvomnibench /path/to/LVOmniBench

Each clip is normalized once (30 fps, width <= 640, lossless H.264; 16 kHz mono
float PCM). The exchanged tracks are exact permutations of its decoded frames
and samples, which is verified on the finished files. The original version
passes through the same pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from .cases import PARTNER, VERSIONS, Case, load_cases

FPS, SAMPLE_RATE = 30, 16000
ENCODE = [
    "-r", str(FPS), "-fps_mode", "cfr", "-enc_time_base", f"1:{FPS}", "-video_track_timescale", "30000",
    "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-threads", "2", "-an", "-map_metadata", "-1",
]


def run(*command: str) -> bytes:
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def exchange(track: np.ndarray, windows: list[list[float]], rate: int) -> np.ndarray:
    """Swap windows 1<->3 and 2<->4 of a per-frame or per-sample track; everything else stays in place."""
    out = track.copy()
    for window, partner in enumerate(PARTNER):
        start, end = (round(t * rate) for t in windows[window])
        source_start, source_end = (round(t * rate) for t in windows[partner])
        out[start:end] = track[source_start:source_end]
    return out


def frame_hashes(ffmpeg: str, path: Path) -> np.ndarray:
    lines = run(ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-f", "framemd5", "-").decode().splitlines()
    return np.asarray([line.rsplit(",", 1)[1].strip() for line in lines if line and not line.startswith("#")])


def decode_audio(ffmpeg: str, path: Path, seek: tuple[str, ...] = (), limit: tuple[str, ...] = ()) -> np.ndarray:
    raw = run(ffmpeg, "-v", "error", *seek, "-i", str(path), *limit, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", "-f", "f32le", "-")
    return np.frombuffer(raw, dtype="<f4").copy()


def render(case: Case, source: Path, out_dir: Path, ffmpeg: str = "ffmpeg") -> None:
    import soundfile

    out_dir.mkdir(parents=True, exist_ok=True)
    duration, windows = float(case["duration"]), case["windows"]
    num_frames, num_samples = round(duration * FPS), round(duration * SAMPLE_RATE)
    seek, limit = ("-ss", str(float(case["offset"]))), ("-t", str(duration))
    video = [out_dir / "video_original.mp4", out_dir / "video_exchanged.mp4"]
    audio = [out_dir / "audio_original.wav", out_dir / "audio_exchanged.wav"]

    run(ffmpeg, "-v", "error", *seek, "-i", str(source), *limit, "-map", "0:v:0",
        "-vf", f"fps={FPS},scale=w='min(640,iw)':h=-2,setpts=PTS-STARTPTS", *ENCODE, "-y", str(video[0]))
    samples = decode_audio(ffmpeg, source, seek, limit)
    if len(samples) != num_samples:
        raise RuntimeError(f"{source}: decoded {len(samples)} audio samples, expected {num_samples}")
    tracks = [samples, exchange(samples, windows, SAMPLE_RATE)]
    for path, track in zip(audio, tracks):
        soundfile.write(path, track, SAMPLE_RATE, subtype="FLOAT")

    # Cut the clip into gaps and windows, then concatenate with the windows exchanged.
    edges = [0.0, *(t for window in windows for t in window), duration]
    segments = list(zip(edges[:-1], edges[1:]))
    present = [i for i, (start, end) in enumerate(segments) if end > start]
    filters = [f"[0:v]split={len(present)}" + "".join(f"[p{i}]" for i in present)]
    filters += [
        f"[p{i}]trim=start_frame={round(segments[i][0] * FPS)}:end_frame={round(segments[i][1] * FPS)},setpts=PTS-STARTPTS[s{i}]"
        for i in present
    ]
    order = [0, 5, 2, 7, 4, 1, 6, 3, 8]
    filters.append("".join(f"[s{i}]" for i in order if i in present) + f"concat=n={len(present)}:v=1:a=0[out]")
    run(ffmpeg, "-v", "error", "-i", str(video[0]), "-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
        "-map", "[out]", *ENCODE, "-y", str(video[1]))

    frames = frame_hashes(ffmpeg, video[0])
    if len(frames) != num_frames or not np.array_equal(frame_hashes(ffmpeg, video[1]), exchange(frames, windows, FPS)):
        raise RuntimeError(f"{case['id']}: exchanged video is not an exact frame permutation")
    expected_frames = [frames, exchange(frames, windows, FPS)]

    for version in VERSIONS:
        a, v = int(version in ("A", "AV")), int(version in ("V", "AV"))
        path = out_dir / f"{version}.mov"
        run(ffmpeg, "-v", "error", "-i", str(video[v]), "-i", str(audio[a]), "-map", "0:v:0", "-map", "1:a:0",
            "-c", "copy", "-video_track_timescale", "30000", "-map_metadata", "-1", "-y", str(path))
        if not (np.array_equal(decode_audio(ffmpeg, path), tracks[a]) and np.array_equal(frame_hashes(ffmpeg, path), expected_frames[v])):
            raise RuntimeError(f"{path}: finished media differs from the intended tracks")
    for path in (*video, *audio):
        path.unlink()


def find_source(root: Path, video_id: str) -> Path:
    matches = [path for pattern in (f"{video_id}.mp4", f"{video_id}_video.mp4") for path in root.rglob(pattern)]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one source file for {video_id} under {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="data/swap_test/cases.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--daily-omni", required=True, type=Path, help="directory holding the Daily-Omni videos")
    parser.add_argument("--lvomnibench", required=True, type=Path, help="directory holding the LVOmniBench videos")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    roots = {"Daily-Omni": args.daily_omni, "LVOmniBench": args.lvomnibench}
    for case in load_cases(args.cases):
        out_dir = args.output_dir / case["id"]
        if not all((out_dir / f"{version}.mov").exists() for version in VERSIONS):
            render(case, find_source(roots[case["benchmark"]], case["video_id"]), out_dir, args.ffmpeg)
            print(case["id"], flush=True)


if __name__ == "__main__":
    main()
