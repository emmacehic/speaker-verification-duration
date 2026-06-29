
"""
Prepare VoxCeleb1 audio for the speaker-duration experiment.

The program:
1. Reads the official verification pairs.
2. Keeps pairs whose second recording is at least 10 seconds.
3. Optionally selects a smaller balanced group.
4. Saves full reference recordings.
5. Saves 3, 5, and 10-second test recordings.
6. Creates selected_trials.txt and manifest.csv.
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


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "voxceleb1"
AUDIO_ROOT = DATA_ROOT / "wav"
TRIAL_FILE = DATA_ROOT / "veri_test2.txt"

# Output folders and files
OUTPUT_ROOT = DATA_ROOT / "prepared_audio"
REFERENCE_ROOT = OUTPUT_ROOT / "reference"

TEST_ROOTS = {
    3: OUTPUT_ROOT / "test_3s",
    5: OUTPUT_ROOT / "test_5s",
    10: OUTPUT_ROOT / "test_10s",
}

# These two files describe exactly which trials and prepared audio paths were made.
SELECTED_TRIALS_FILE = OUTPUT_ROOT / "selected_trials.txt"
MANIFEST_FILE = OUTPUT_ROOT / "manifest.csv"

# Audio settings
# All saved recordings use the same sample rate so the later experiment is consistent.
TARGET_SAMPLE_RATE = 16000
DURATIONS = (3, 5, 10)


# This function converts the official text protocol into structured trial dictionaries.

def read_trials() -> list[dict]:

    """Read valid verification pairs from veri_test2.txt."""


    trials = []

    # UTF-8 is used explicitly so the input is read consistently on different systems.
    with TRIAL_FILE.open("r", encoding="utf-8") as file:

        
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            
            if not line:
                continue

            parts = line.split()

            # Incorrectly formatted rows are skipped instead of stopping the whole preparation.
            if len(parts) != 3:
                print(
                    f"Skipping line {line_number}: "
                    "expected 3 columns."
                )
                continue

            label_text, first_file, second_file = parts

            # The label must be converted from file text into an integer class value.
            try:
                label = int(label_text)
            except ValueError:
                print(
                    f"Skipping line {line_number}: "
                    "invalid label."
                )
                continue

            # VoxCeleb verification uses 1 for a match and 0 for different speakers.
            if label not in (0, 1):
                print(
                    f"Skipping line {line_number}: "
                    "label must be 0 or 1."
                )
                continue

            # Relative names and complete paths are being stored.
            trials.append(
                {
                    "label": label,
                    "first_file": first_file,
                    "second_file": second_file,
                    "first_path": AUDIO_ROOT / first_file,
                    "second_path": AUDIO_ROOT / second_file,
                }
            )

    return trials



def duration_seconds(path: Path) -> float:

    """Return the duration of an audio file in seconds."""

    # SoundFile provides the frame count and sample rate directly from the WAV header.
    info = sf.info(path)

    # Dividing frames by frames per second gives the duration in seconds.
    return info.frames / info.samplerate


def load_mono_16k(path: Path) -> np.ndarray:

    """
    Load audio, convert it to mono, and resample
    it to 16 kHz.

    """

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    # Average all channels to make one mono channel.
    # Taking the channel mean is a simple way to produce one mono waveform.
    audio = audio.mean(axis=1)

    # Resamples only when the original rate is not 16 kHz.
    # Files already recorded at the target rate can skip this extra processing step.
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE,
            sample_rate,
        ).astype("float32")

    
    return audio


def save_audio(path: Path, audio: np.ndarray) -> None:

    """Save audio as a 16 kHz, 16-bit PCM WAV file."""

    # Creating parent folders automatically because speaker files use nested directories.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keeping all files consistent.
    sf.write(
        path,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )


# This filtering stage removes trials that cannot support all three duration conditions.

def select_usable_trials(
    trials: list[dict],
) -> list[dict]:
    
    """
    Keep trials where both files exist and the second file
    is readable and at least 10 seconds long.
    """

    usable = []
    missing = 0
    unreadable = 0
    too_short = 0

    # The progress bar is helpful because checking a large dataset can take some time.
    for trial in tqdm(
        trials,
        desc="Checking official trials",
    ):
        first_path = trial["first_path"]
        second_path = trial["second_path"]

        # Both sides of a verification pair must exist before the trial can be used.
        if (
            not first_path.is_file()
            or not second_path.is_file()
        ):
            missing += 1
            continue

        # Metadata reading is protected because a present file may still be damaged or inaccessible.
        try:
            second_duration = duration_seconds(
                second_path
            )
        except Exception:
            unreadable += 1
            continue

        # Ten seconds is the longest test condition.
        # A file shorter than the maximum duration cannot be used fairly in every condition.
        if second_duration < max(DURATIONS):
            too_short += 1
            continue

        # Copy the trial before adding its duration.
        usable_trial = dict(trial)

        # The measured duration is kept for the manifest created later.
        usable_trial["second_duration"] = (
            second_duration
        )

        usable.append(usable_trial)

    print(
        f"Trials: {len(trials)} total, "
        f"{len(usable)} usable, "
        f"{missing} missing, "
        f"{unreadable} unreadable, "
        f"{too_short} under 10 seconds."
    )

    return usable


def balanced_limit(
    trials: list[dict],
    limit: int | None,
) -> list[dict]:
    
    """
    Return all trials or a smaller approximately
    balanced group.
    """

    if limit is None or limit >= len(trials):
        return trials

    # Same-speaker trials are separated first so the requested subset can be balanced.
    same_speaker = [
        trial
        for trial in trials
        if trial["label"] == 1
    ]

    # Label zero trials form the second group which is used in the selection.
    different_speakers = [
        trial
        for trial in trials
        if trial["label"] == 0
    ]

    # With an odd limit, different-speaker trials get one extra place.
    # Integer division gives half the places to same-speaker examples.
    same_needed = limit // 2
    different_needed = limit - same_needed

    selected = (
        same_speaker[:same_needed]
        + different_speakers[:different_needed]
    )

    # Fill empty places if one group did not contain enough trials.
    
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

            # Only trials that are not already in the subset can fill an empty place.
            if key not in already_selected:
                selected.append(trial)
                already_selected.add(key)

            if len(selected) == limit:
                break

    return selected


def prepare_audio(trials: list[dict]) -> None:

    """
    Create the reference files, cropped test files,
    selected trial list, and CSV manifest.
    """

    
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remembering which files were already prepared.
    # This prevents repeated recordings from being processed more than once.
    prepared_references: set[str] = set()

    prepared_tests: dict[int, set[str]] = {
        duration: set()
        for duration in DURATIONS
    }

    manifest_rows = []

    for trial_number, trial in enumerate(
        tqdm(
            trials,
            desc="Preparing audio",
        ),
        start=1,
    ):
        # Keeping the relative dataset path preserves the original speaker folder structure.
        first_relative = Path(
            trial["first_file"]
        )

        # The test recording uses the same relative-path approach as the reference recording.
        second_relative = Path(
            trial["second_file"]
        )

        reference_output = (
            REFERENCE_ROOT / first_relative
        )

        # Preparing each reference recording only once.
        # Repeated reference files can reuse the first prepared copy.
        if (
            trial["first_file"]
            not in prepared_references
        ):
            # The full reference is converted to mono 16 kHz without being cropped.
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

        # Loading the test recording only if a new clip needs to be created.
        # The full test waveform is read only when a clip is actually missing.
        second_audio = None
        
        test_outputs = {}

        # The same source test recording is prepared at every experiment duration.
        for duration in DURATIONS:
            
            test_output = (
                TEST_ROOTS[duration]
                / second_relative
            )

            test_outputs[duration] = test_output

            # A previously prepared clip can be skipped even when it appears in another trial.
            if (
                trial["second_file"]
                in prepared_tests[duration]
            ):
                continue

            # The source test recording is loaded once and reused for all required crops.
            if second_audio is None:
                second_audio = load_mono_16k(
                    trial["second_path"]
                )

            # Convert seconds to the required number of audio samples.
            # Multiplying seconds by samples per second gives the exact crop length.
            sample_count = (
                duration * TARGET_SAMPLE_RATE
            )

            # Keeping the beginning of the recording, so that the experiment consistently uses the beginning of each test recording.
            clip = second_audio[:sample_count]

            
            # This check guards against unexpected problems after resampling or file changes.
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

            # Updating the cache.
            prepared_tests[duration].add(
                trial["second_file"]
            )

        # Saveing one manifest row for this trial.
        # The manifest links the original pair with every generated version of the audio.
        manifest_rows.append(
            {
                "trial_number": trial_number,
                "label": trial["label"],

                # A readable label description is included beside the numeric class.
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


    # Saveing the exact trial pairs used.
    # Files are being opened once, because it's more efficient than reopening it for every trial.
    with SELECTED_TRIALS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        
        for trial in trials:
            file.write(
                f"{trial['label']} "
                f"{trial['first_file']} "
                f"{trial['second_file']}\n"
            )

    # Saveing the manifest using the same columns and order as the original program.
   
    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # Counting both classes provides a final check of the selected trial balance.
    same_count = sum(
        trial["label"] == 1
        for trial in trials
    )

    # The different-speaker total is calculated separately for a clearer completion message.
    different_count = sum(
        trial["label"] == 0
        for trial in trials
    )

    print(
        f"Prepared {len(trials)} trials: "
        f"{same_count} same-speaker and "
        f"{different_count} different-speaker."
    )

    print(
        f"Files saved in: {OUTPUT_ROOT}"
    )



def main() -> None:

    """Read the command-line options and prepare the data."""

    parser = argparse.ArgumentParser(
        description=(
            "Select and prepare VoxCeleb1 "
            "verification audio."
        )
    )

    # The limit option for a small test run before processing the full dataset, so no time is wasted.
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional balanced number of trials "
            "for a small test. Leave this out "
            "to use every usable trial."
        ),
    )

    # Only optional clearing, so that existing prepared files are not deleted by accident.
    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Delete the old prepared_audio folder "
            "before starting."
        ),
    )

    
    args = parser.parse_args()

    # A balanced experiment needs at least two trials.
    if (
        args.limit is not None
        and args.limit < 2
    ):
        raise SystemExit(
            "--limit must be at least 2."
        )

    # The program stopping early with a useful path if the main audio folder is missing.
    if not AUDIO_ROOT.is_dir():
        raise SystemExit(
            "Audio folder not found.\n"
            f"Expected: {AUDIO_ROOT}\n"
            "Put the id10270, id10271, ... "
            "folders inside that wav folder."
        )

    # The official pair list is also required before any trial selection can happen.
    if not TRIAL_FILE.is_file():
        raise SystemExit(
            "Verification file not found.\n"
            f"Expected: {TRIAL_FILE}"
        )

    # Old prepared files can be deleted, but only when --clear is used.
    if (
        args.clear
        and OUTPUT_ROOT.exists()
    ):
        shutil.rmtree(OUTPUT_ROOT)

    # First, the official protocol is parsed and invalid text rows are removed.
    all_trials = read_trials()

    # Then unavailable and short audio pairs are filtered out.
    usable_trials = select_usable_trials(
        all_trials
    )

    selected_trials = balanced_limit(
        usable_trials,
        args.limit,
    )

    # Preventing an error, by not continuing with an empty list.
    if not selected_trials:
        raise SystemExit(
            "No usable trials were found. "
            "Make sure the iCloud files "
            "are fully downloaded."
        )

    # Both label groups are needed.
    labels = {
        trial["label"]
        for trial in selected_trials
    }

    if labels != {0, 1}:
        raise SystemExit(
            "The selected trials do not contain "
            "both labels 0 and 1. "
            "Try a larger --limit value."
        )

  
    prepare_audio(selected_trials)



if __name__ == "__main__":
    main()