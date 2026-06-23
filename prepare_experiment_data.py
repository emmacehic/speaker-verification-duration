"""
prepare_experiment_data.py

Run this file from the main project folder:

    cd /path/to/speaker-duration-project
    python prepare_experiment_data.py

What it does:

1. Reads data/voxceleb1/veri_test2.txt.
2. Finds the official verification pairs.
3. Keeps only pairs whose second recording is at least 10 seconds long.
4. Saves the exact selected pairs.
5. Creates prepared 3-, 5-, and 10-second test clips.
6. Copies each required reference recording once.
7. Writes a CSV manifest describing every selected pair.

Nothing is selected manually.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm



# Project folders and preparation settings


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "voxceleb1"
AUDIO_ROOT = DATA_ROOT / "wav"
TRIAL_FILE = DATA_ROOT / "veri_test2.txt"

OUTPUT_ROOT = DATA_ROOT / "prepared_audio"
REFERENCE_ROOT = OUTPUT_ROOT / "reference"

TEST_ROOTS = {
    3: OUTPUT_ROOT / "test_3s",
    5: OUTPUT_ROOT / "test_5s",
    10: OUTPUT_ROOT / "test_10s",
}

SELECTED_TRIALS_FILE = OUTPUT_ROOT / "selected_trials.txt"
MANIFEST_FILE = OUTPUT_ROOT / "manifest.csv"

TARGET_SAMPLE_RATE = 16000
DURATIONS = (3, 5, 10)



# Read the VoxCeleb verification-pair file


def read_trials() -> list[dict]:
    """
    Read all verification pairs from veri_test2.txt.

    Each line should contain:

    label first_recording.wav second_recording.wav

    A label of 1 means that both recordings contain
    the same speaker.

    A label of 0 means that the recordings contain
    different speakers.

    Invalid lines are skipped and a message is printed.
    """

    trials = []

    with TRIAL_FILE.open("r", encoding="utf-8") as handle:

        for line_number, raw_line in enumerate(handle, start=1):

            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                print(
                    f"Skipping line {line_number}: "
                    "expected 3 columns."
                )
                continue

            label_text, first_file, second_file = parts

            try:
                label = int(label_text)
            except ValueError:
                print(
                    f"Skipping line {line_number}: "
                    "invalid label."
                )
                continue

            if label not in (0, 1):
                print(
                    f"Skipping line {line_number}: "
                    "label must be 0 or 1."
                )
                continue

            trial = {
                "label": label,
                "first_file": first_file,
                "second_file": second_file,
                "first_path": AUDIO_ROOT / first_file,
                "second_path": AUDIO_ROOT / second_file,
            }

            trials.append(trial)

    return trials



# Find the duration of an audio recording


def duration_seconds(path: Path) -> float:
    """
    Return the duration of an audio file in seconds.

    Only the file information is read, so the complete
    audio recording does not need to be loaded into memory.
    """

    info = sf.info(path)

    duration = info.frames / info.samplerate

    return duration



# Load and standardise an audio recording


def load_mono_16k(path: Path) -> np.ndarray:
    """
    Load an audio recording as a NumPy array.

    The recording is prepared in two ways:

    - Stereo or multichannel audio is changed to mono.
    - Audio is resampled to 16 kHz when necessary.

    The returned audio uses 32-bit floating-point values.
    """

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    # Average all channels to create a mono recording.
    audio = audio.mean(axis=1)

    # The speaker-verification model expects 16 kHz audio.
    if sample_rate != TARGET_SAMPLE_RATE:

        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE,
            sample_rate,
        ).astype("float32")

    return audio



# Save a prepared audio recording


def save_audio(path: Path, audio: np.ndarray) -> None:
    """
    Save an audio array as a 16 kHz WAV file.

    Any missing parent folders are created automatically.
    The file is saved using 16-bit PCM encoding.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        path,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )



# Select the trials that can be prepared


def select_usable_trials(
    trials: list[dict],
) -> list[dict]:
    """
    Keep only verification trials that can be used.

    A trial is usable when:

    - Both audio files exist.
    - The second recording can be read.
    - The second recording is at least 10 seconds long.

    The duration of the second recording is added to each
    usable trial so that it can later be written to the
    manifest file.
    """

    usable = []

    missing = 0
    unreadable = 0
    too_short = 0

    for trial in tqdm(
        trials,
        desc="Checking official trials",
    ):

        first_path = trial["first_path"]
        second_path = trial["second_path"]

        # Both recordings are required for verification.
        if (
            not first_path.is_file()
            or not second_path.is_file()
        ):
            missing += 1
            continue

        try:
            second_duration = duration_seconds(
                second_path
            )
        except Exception:
            unreadable += 1
            continue

        # Ten seconds is the longest experimental condition.
        if second_duration < max(DURATIONS):
            too_short += 1
            continue

        # Copy the dictionary before adding new information.
        trial = dict(trial)

        trial["second_duration"] = second_duration

        usable.append(trial)

    print()
    print("DATA SELECTION")
    print("=" * 50)
    print(
        f"Official trial lines:          {len(trials)}"
    )
    print(
        f"Usable for 3/5/10 seconds:    {len(usable)}"
    )
    print(
        f"Missing files:                 {missing}"
    )
    print(
        f"Unreadable files:              {unreadable}"
    )
    print(
        f"Second recording under 10 s:  {too_short}"
    )
    print("=" * 50)

    return usable



# Select a smaller balanced group of trials


def balanced_limit(
    trials: list[dict],
    limit: int | None,
) -> list[dict]:
    """
    Optionally select a smaller, approximately balanced
    number of trials.

    Half of the selected trials are taken from the
    same-speaker group and half from the different-speaker
    group.

    When no limit is given, all usable trials are returned.
    """

    if limit is None or limit >= len(trials):
        return trials

    same = [
        trial
        for trial in trials
        if trial["label"] == 1
    ]

    different = [
        trial
        for trial in trials
        if trial["label"] == 0
    ]

    # For an odd limit, the different-speaker group receives
    # one more trial than the same-speaker group.
    same_needed = limit // 2
    different_needed = limit - same_needed

    selected = (
        same[:same_needed]
        + different[:different_needed]
    )

    # This section fills any remaining places if one label
    # group does not contain enough trials.
    if len(selected) < limit:

        already_selected = {
            (
                trial["label"],
                trial["first_file"],
                trial["second_file"],
            )
            for trial in selected
        }

        for trial in trials:

            key = (
                trial["label"],
                trial["first_file"],
                trial["second_file"],
            )

            if key not in already_selected:

                selected.append(trial)
                already_selected.add(key)

            if len(selected) == limit:
                break

    return selected



# Prepare the reference and test audio files


def prepare_audio(
    trials: list[dict],
) -> None:
    """
    Prepare all audio files needed for the experiment.

    For each verification pair:

    - The complete first recording is saved as the reference.
    - The first 3 seconds of the second recording are saved.
    - The first 5 seconds of the second recording are saved.
    - The first 10 seconds of the second recording are saved.

    Repeated recordings are prepared only once.

    The function also writes:

    - selected_trials.txt
    - manifest.csv
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # These sets keep track of recordings that have already
    # been prepared. This avoids processing duplicate files.
    prepared_references: set[str] = set()

    prepared_tests: dict[int, set[str]] = {
        duration: set()
        for duration in DURATIONS
    }

    # Each row will later become one line in manifest.csv.
    rows = []

    for trial_number, trial in enumerate(
        tqdm(
            trials,
            desc="Preparing audio",
        ),
        start=1,
    ):

        first_relative = Path(
            trial["first_file"]
        )

        second_relative = Path(
            trial["second_file"]
        )

        reference_output = (
            REFERENCE_ROOT / first_relative
        )

        # Prepare each reference recording only once.
        if (
            trial["first_file"]
            not in prepared_references
        ):

            reference_audio = load_mono_16k(
                trial["first_path"]
            )

            save_audio(
                reference_output,
                reference_audio,
            )

            prepared_references.add(
                trial["first_file"]
            )

        # The second recording is loaded only when one of its
        # prepared versions still needs to be created.
        second_audio = None

        test_outputs = {}

        for duration in DURATIONS:

            test_output = (
                TEST_ROOTS[duration]
                / second_relative
            )

            test_outputs[duration] = test_output

            if (
                trial["second_file"]
                not in prepared_tests[duration]
            ):

                if second_audio is None:

                    second_audio = load_mono_16k(
                        trial["second_path"]
                    )

                # Convert the requested duration from seconds
                # to the required number of audio samples.
                sample_count = (
                    duration
                    * TARGET_SAMPLE_RATE
                )

                # Keep the beginning of the recording.
                clip = second_audio[:sample_count]

                # This should not occur because the duration
                # was checked earlier.
                if len(clip) < sample_count:

                    raise RuntimeError(
                        "Unexpected short file "
                        "after checking: "
                        f"{trial['second_path']}"
                    )

                save_audio(
                    test_output,
                    clip,
                )

                prepared_tests[duration].add(
                    trial["second_file"]
                )

        # Record the original and prepared file locations.
        rows.append(
            {
                "trial_number": trial_number,
                "label": trial["label"],
                "label_meaning": (
                    "same speaker"
                    if trial["label"] == 1
                    else "different speakers"
                ),
                "original_reference": (
                    trial["first_file"]
                ),
                "original_test": (
                    trial["second_file"]
                ),
                "original_test_duration_seconds": round(
                    trial["second_duration"],
                    4,
                ),
                "prepared_reference": str(
                    reference_output.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "prepared_test_3s": str(
                    test_outputs[3].relative_to(
                        PROJECT_ROOT
                    )
                ),
                "prepared_test_5s": str(
                    test_outputs[5].relative_to(
                        PROJECT_ROOT
                    )
                ),
                "prepared_test_10s": str(
                    test_outputs[10].relative_to(
                        PROJECT_ROOT
                    )
                ),
            }
        )

    # Save the exact official trial pairs used.
    with SELECTED_TRIALS_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for trial in trials:

            handle.write(
                f"{trial['label']} "
                f"{trial['first_file']} "
                f"{trial['second_file']}\n"
            )

    # Save a table describing every prepared trial.
    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("PREPARATION COMPLETE")
    print("=" * 50)
    print(
        f"Selected trials: {len(trials)}"
    )
    print(
        "Same-speaker trials:",
        sum(
            trial["label"] == 1
            for trial in trials
        ),
    )
    print(
        "Different-speaker trials:",
        sum(
            trial["label"] == 0
            for trial in trials
        ),
    )
    print(
        "Prepared audio folder:"
    )
    print(OUTPUT_ROOT)
    print(
        "Trial list:"
    )
    print(SELECTED_TRIALS_FILE)
    print(
        "Manifest:"
    )
    print(MANIFEST_FILE)
    print("=" * 50)



# Main part of the program


def main() -> None:
    """
    Start the data-preparation process.

    To prepare all usable official trials, run:

        python prepare_experiment_data.py

    To prepare a smaller balanced test set, run:

        python prepare_experiment_data.py --limit 20

    To remove previously prepared files before starting, run:

        python prepare_experiment_data.py --clear
    """

    parser = argparse.ArgumentParser(
        description=(
            "Select and prepare VoxCeleb1 "
            "verification audio."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional balanced number of trials "
            "for a small test. Omit this option "
            "to prepare every usable official trial."
        ),
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Delete the old prepared_audio folder "
            "before starting."
        ),
    )

    args = parser.parse_args()

    print(
        "Project folder:",
        PROJECT_ROOT,
    )
    print(
        "Audio folder:",
        AUDIO_ROOT,
    )
    print(
        "Trial file:",
        TRIAL_FILE,
    )
    print()

    # A limit below two cannot contain both label groups.
    if (
        args.limit is not None
        and args.limit < 2
    ):
        raise SystemExit(
            "--limit must be at least 2."
        )

    if not AUDIO_ROOT.is_dir():

        raise SystemExit(
            "\nAudio folder not found.\n"
            f"Expected: {AUDIO_ROOT}\n"
            "Put the id10270, id10271, ... "
            "folders inside that wav folder."
        )

    if not TRIAL_FILE.is_file():

        raise SystemExit(
            "\nVerification file not found.\n"
            f"Expected: {TRIAL_FILE}"
        )

    # Remove previous prepared files only when the user
    # explicitly includes the --clear option.
    if (
        args.clear
        and OUTPUT_ROOT.exists()
    ):

        print(
            "Removing old prepared audio..."
        )

        shutil.rmtree(OUTPUT_ROOT)

    print(
        "Reading the official verification pairs..."
    )

    all_trials = read_trials()

    usable_trials = select_usable_trials(
        all_trials
    )

    selected_trials = balanced_limit(
        usable_trials,
        args.limit,
    )

    if not selected_trials:

        raise SystemExit(
            "\nNo usable trials were found. "
            "Make sure the iCloud files "
            "are fully downloaded."
        )

    # Both same-speaker and different-speaker trials are
    # needed for a speaker-verification experiment.
    labels = {
        trial["label"]
        for trial in selected_trials
    }

    if labels != {0, 1}:

        raise SystemExit(
            "\nThe selected trials do not contain "
            "both labels 0 and 1. "
            "Try a larger --limit value."
        )

    prepare_audio(
        selected_trials
    )


if __name__ == "__main__":
    main()