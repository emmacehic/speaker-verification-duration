
"""
Test how 3, 5, and 10-second test recordings affect
speaker-verification performance.

The first recording in each pair stays unchanged.
The prepared 3, 5, and 10-second test recordings are loaded.
The program compares speaker embeddings with cosine similarity
and calculates the Equal Error Rate (EER).
"""

import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from scipy.signal import resample_poly
from sklearn.metrics import roc_curve
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

# The experiment uses the files made by prepare_experiment_data.py not the original WAV files.
PREPARED_AUDIO_FOLDER = Path(
    "data/voxceleb1/prepared_audio"
)

REFERENCE_FOLDER = (
    PREPARED_AUDIO_FOLDER / "reference"
)

# Each duration has its own folder of prepared test recordings.
TEST_FOLDERS = {
    3: PREPARED_AUDIO_FOLDER / "test_3s",
    5: PREPARED_AUDIO_FOLDER / "test_5s",
    10: PREPARED_AUDIO_FOLDER / "test_10s",
}

# Useing the exact trial list created by prepare_experiment_data.py.
TRIAL_FILE = (
    PREPARED_AUDIO_FOLDER / "selected_trials.txt"
)

RESULTS_FOLDER = Path("results")
MODEL_FOLDER = Path("saved_models/ecapa_tdnn")


# Experiment settings.
# The pretrained model expects 16 kHz speech, so all audio is brought to this rate.
SAMPLE_RATE = 16000

# These are the three test conditions that will be compared in the results.
TEST_DURATIONS = [3, 5, 10]



# The first stage is to turn the verification protocol into usable Python records.

def read_trials():

    """Read the speaker-verification pairs from the trial file."""

    # Each valid line from the protocol will become one dictionary in this list.

    trials = []

    # Reading line by line also lets the program report the exact location of bad input.
    with TRIAL_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            # A correct trial line contains one label and two relative audio-file names.
            parts = line.split()

            if len(parts) != 3:

                print(f"Skipping line {line_number}: wrong format.")

                continue

            label_text, first_file, second_file = parts

            # Labels are being stored as text in the file, so they need to be converted first.
            try:
                label = int(label_text)

            except ValueError:
                print(f"Skipping line {line_number}: invalid label.")

                continue

            # Only the two expected verification classes are accepted.
            if label not in [0, 1]:
                print(
                    f"Skipping line {line_number}: "
                    "label must be 0 or 1.")
                
                continue

           # Connecting the trial to the prepared reference file and the three prepared test files.
            trials.append(
                {
                    "label": label,
                    "first_file": first_file,
                    "second_file": second_file,

                    # Full prepared reference recording.
                    "first_path": (
                        REFERENCE_FOLDER
                        / first_file),

                    # Prepared test recording for each duration.
                    "test_paths": {
                        duration: (TEST_FOLDERS[duration]
                            / second_file)

                        for duration in TEST_DURATIONS},})

    return trials



def select_usable_trials(all_trials):

    """
    Keep trials whose prepared reference and test recordings all exist.

    """

    usable_trials = []

    missing_files = 0
   

    # prepare_experiment_data.py already checked whether
    # the original test recording was at least 10 seconds.
    # This script only needs to check whether the prepared files exist.
    
    for trial in tqdm(
        all_trials,
        desc="Checking prepared files",
    ):
        required_files = [
            trial["first_path"],
            trial["test_paths"][3],
            trial["test_paths"][5],
            trial["test_paths"][10],
        ]

        if not all(
            path.is_file()
            for path in required_files):
            
            missing_files += 1
            continue

        usable_trials.append(trial)

    print(
        f"Trials: {len(all_trials)} total, "
        f"{len(usable_trials)} usable, "
        f"{missing_files} missing prepared files."
    )

    return usable_trials


def load_audio(audio_path):

    """
    Load audio, convert it to mono, and resample it
    to 16 kHz if needed.
    """

    audio, original_sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    # Averageing all channels to create one mono channel.
    audio = audio.mean(axis=1)

    # Resampling keeps the sample counts consistent across all source recordings.
    if original_sample_rate != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            original_sample_rate,
        )

        audio = audio.astype("float32")

    return torch.tensor(audio, dtype=torch.float32)


def extract_embedding(model, waveform):

    """Create one normalized speaker embedding."""

    # The model expects a batch.
    batch = waveform.unsqueeze(0)

    with torch.inference_mode():
        embedding = model.encode_batch(
            batch,
            normalize=True,
        )

    # The extra dimensions from batching are removed to keep one embedding vector.
    embedding = embedding.squeeze()

    # Normalizing again keeps the vector length equal to one for cosine comparison.
    embedding = F.normalize(embedding, dim=0)

    return embedding.cpu()


def calculate_cosine_similarity(
    first_embedding,
    second_embedding,
):
    """Return the cosine similarity between two embeddings."""
    

    score = F.cosine_similarity(
        first_embedding.unsqueeze(0),
        second_embedding.unsqueeze(0),
    )

    return float(score.item())


def calculate_eer(labels, scores):

    """Calculate the Equal Error Rate and its threshold."""
    
    false_positive_rates, true_positive_rates, thresholds = (
        roc_curve(
            labels,
            scores,
            pos_label=1,
        )
    )

    # False-negative rate is calculated from the true-positive rate returned by sklearn.
    false_negative_rates = 1 - true_positive_rates

    # Finding where false positives and false negatives are closest to each other.
    differences = np.abs(
        false_positive_rates - false_negative_rates
    )

    # The closest crossing point is used as a practical estimate of the EER position.
    closest_index = np.argmin(differences)

    # Averaging the two error rates gives one balanced error value at this threshold.
    eer = (
        false_positive_rates[closest_index]
        + false_negative_rates[closest_index]
    ) / 2

    threshold = thresholds[closest_index]

    return float(eer), float(threshold)


def save_selected_trials(trials):

    """Save the exact trial pairs used in the experiment."""

    output_file = RESULTS_FOLDER / "selected_trials.txt"

    
    with output_file.open("w", encoding="utf-8") as file:
        for trial in trials:
            file.write(
                f"{trial['label']} "
                f"{trial['first_file']} "
                f"{trial['second_file']}\n"
            )



def run_experiment(model, trials):

    """
    Run every trial using 3, 5, and 10-second
    test recordings.
    """

    
    all_results = []

    first_embedding_cache = {}

    test_embedding_cache = {}

    # The full set of trials is repeated once for each recording-duration condition.
    for duration in TEST_DURATIONS:
        for trial_number, trial in enumerate(
            tqdm(trials, desc=f"{duration}-second trials",),start=1,):

            first_path = trial["first_path"]

             # Selecting the prepared test recording belonging to the current duration.
            second_path = trial["test_paths"][duration]

            first_key = str(first_path)

            second_key = str(second_path)

            # The first recording never changes, so its embedding can be reused.
            if first_key not in first_embedding_cache:
                first_waveform = load_audio(first_path)

                first_embedding_cache[first_key] = (
                    extract_embedding(model, first_waveform,))

            # The cached enrollment embedding is reused for every matching trial entry.
            first_embedding = first_embedding_cache[first_key]

            # Loading the already cropped test file.
            if second_key not in test_embedding_cache:

                second_waveform = load_audio(second_path)

                test_embedding_cache[second_key] = (
                    extract_embedding(
                        model,
                        second_waveform,))

            second_embedding = (
                test_embedding_cache[
                    second_key])

            # This score is the model output used to decide whether the speakers match.
            similarity_score = calculate_cosine_similarity(
                first_embedding,
                second_embedding,)

            # Keeping identifiers beside the score.
            all_results.append(
                {
                    "trial_number": trial_number,
                    "duration_seconds": duration,
                    "label": trial["label"],
                    "first_file": trial["first_file"],
                    "second_file": trial["second_file"],
                    "cosine_similarity": similarity_score,
                }
            )

    results_dataframe = pd.DataFrame(all_results)

    results_dataframe.to_csv(
        RESULTS_FOLDER / "trial_scores.csv",
        index=False,)

    return results_dataframe


def create_summary(results_dataframe):

    """
    Calculate EER and score statistics for each
    recording duration.
    """

    summary_rows = []

    # One summary row is produced for every duration in the experiment.

    for duration in TEST_DURATIONS:

        # The calculations below use only the trials from the current duration.

        duration_results = results_dataframe[
            results_dataframe["duration_seconds"] == duration
        ]

        # NumPy arrays are used because it needs array-like values.
        labels = duration_results["label"].to_numpy()

        scores = duration_results[
            "cosine_similarity"
        ].to_numpy()

        eer, threshold = calculate_eer(
            labels,
            scores,
        )

        # The two classes are separated so their score behaviour can be compared.
        same_speaker_results = duration_results[
            duration_results["label"] == 1
        ]

        different_speaker_results = duration_results[
            duration_results["label"] == 0
        ]

        # Mean scores give a simple overall centre for the same-speaker group.
        mean_same_score = same_speaker_results[
            "cosine_similarity"
        ].mean()

        # The same calculation is repeated for the different-speaker group.
        mean_different_score = different_speaker_results[
            "cosine_similarity"
        ].mean()

        # Medians are included because they are less sensitive to unusual score values.
        median_same_score = same_speaker_results[
            "cosine_similarity"
        ].median()

        median_different_score = different_speaker_results[
            "cosine_similarity"
        ].median()

        # Combining performance metrics and descriptive statistics.
        summary_rows.append(
            {
                "duration_seconds": duration,
                "number_of_trials": len(duration_results),
                
                "same_speaker_trials": len(
                    same_speaker_results
                ),
                "different_speaker_trials": len(
                    different_speaker_results
                ),
                "eer": eer,
                "eer_percent": eer * 100,
                "eer_threshold": threshold,
                "mean_same_speaker_score": (
                    mean_same_score
                ),
                "mean_different_speaker_score": (
                    mean_different_score
                ),
                "mean_score_difference": (
                    mean_same_score
                    - mean_different_score
                ),
                "median_same_speaker_score": (
                    median_same_score
                ),
                "median_different_speaker_score": (
                    median_different_score
                ),
                "median_score_difference": (
                    median_same_score
                    - median_different_score
                ),
            }
        )

    # The list of summary dictionaries becomes the final compact results table.
    summary_dataframe = pd.DataFrame(summary_rows)

    summary_dataframe.to_csv(
        RESULTS_FOLDER / "summary_results.csv",
        index=False,
    )

    print(summary_dataframe.to_string(index=False))

    return summary_dataframe


def create_eer_graph(summary_dataframe):

    """Create the EER line graph."""


    plt.figure(figsize=(7, 5))

    plt.plot(
        summary_dataframe["duration_seconds"],
        summary_dataframe["eer_percent"],
        marker="o",
    )

    plt.xlabel("Test-recording duration in seconds")
    plt.ylabel("Equal Error Rate (%)")
    plt.title(
        "Effect of Test Duration on Speaker Verification"
    )

    plt.xticks(TEST_DURATIONS)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        RESULTS_FOLDER / "eer_by_duration.png",
        dpi=300,
    )

    plt.close()


def create_average_score_graph(summary_dataframe):

    """
    Create the average same-speaker and
    different-speaker score graph.
    """

    durations = summary_dataframe[
        "duration_seconds"
    ].to_numpy()

    same_scores = summary_dataframe[
        "mean_same_speaker_score"
    ].to_numpy()

    different_scores = summary_dataframe[
        "mean_different_speaker_score"
    ].to_numpy()

    positions = np.arange(len(durations))
    bar_width = 0.35

    plt.figure(figsize=(8, 5))

    # The first bar in each pair represents genuine same-speaker trials.
    same_bars = plt.bar(
        positions - bar_width / 2,
        same_scores,
        width=bar_width,
        label="Same speaker",
    )

    # The second bar shows the corresponding impostor or different-speaker trials.
    different_bars = plt.bar(
        positions + bar_width / 2,
        different_scores,
        width=bar_width,
        label="Different speakers",
    )

    plt.xlabel("Test-recording duration")
    plt.ylabel("Average cosine similarity")
    plt.title("Average Speaker-Verification Scores")

    plt.xticks(
        positions,
        [
            f"{duration} seconds"
            for duration in durations
        ],
    )

    plt.bar_label(
        same_bars,
        fmt="%.3f",
        padding=3,
    )

    plt.bar_label(
        different_bars,
        fmt="%.3f",
        padding=3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_FOLDER / "average_scores.png",
        dpi=300,
    )

    plt.close()


def create_median_score_graph(summary_dataframe):

    """
    Create the median same-speaker and
    different-speaker score graph.
    """

    # This graph follows the same layout as the mean graph but uses median scores.
    durations = summary_dataframe[
        "duration_seconds"
    ].to_numpy()

    same_scores = summary_dataframe[
        "median_same_speaker_score"
    ].to_numpy()

    different_scores = summary_dataframe[
        "median_different_speaker_score"
    ].to_numpy()

    positions = np.arange(len(durations))
    bar_width = 0.35

    plt.figure(figsize=(8, 5))

    same_bars = plt.bar(
        positions - bar_width / 2,
        same_scores,
        width=bar_width,
        label="Same speaker",
    )

    different_bars = plt.bar(
        positions + bar_width / 2,
        different_scores,
        width=bar_width,
        label="Different speakers",
    )

    plt.xlabel("Test-recording duration")
    plt.ylabel("Median cosine similarity")
    plt.title("Median Speaker-Verification Scores")

    plt.xticks(
        positions,
        [
            f"{duration} seconds"
            for duration in durations
        ],
    )

    plt.bar_label(
        same_bars,
        fmt="%.3f",
        padding=3,
    )

    plt.bar_label(
        different_bars,
        fmt="%.3f",
        padding=3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_FOLDER / "median_scores.png",
        dpi=300,
    )

    plt.close()


def create_score_distribution_graphs(
    results_dataframe,
):
    """
    Create one score-distribution graph for each
    test duration.
    """

    graph_labels = {
        3: "(a) 3-second condition",
        5: "(b) 5-second condition",
        10: "(c) 10-second condition",
    }

    # Creating same bins and axes to make the graphs comparable.
    histogram_bins = np.linspace(
        -0.3,
        1.05,
        61,
    )

    # A separate histogram is generated for each recording-length condition.
    for duration in TEST_DURATIONS:
        duration_results = results_dataframe[
            results_dataframe["duration_seconds"]
            == duration
        ]

        # Genuine and impostor scores are plotted separately to show their overlap.
        same_scores = duration_results[
            duration_results["label"] == 1
        ]["cosine_similarity"]

        different_scores = duration_results[
            duration_results["label"] == 0
        ]["cosine_similarity"]

        labels = duration_results[
            "label"
        ].to_numpy()

        scores = duration_results[
            "cosine_similarity"
        ].to_numpy()

        # The duration-specific EER threshold is recalculated for the vertical marker.
        _, eer_threshold = calculate_eer(
            labels,
            scores,
        )

        plt.figure(figsize=(7, 4.5))

        plt.hist(
            same_scores,
            bins=histogram_bins,
            alpha=0.65,
            label="Same speaker",
        )

        plt.hist(
            different_scores,
            bins=histogram_bins,
            alpha=0.65,
            label="Different speakers",
        )

        # The dashed line shows the operating point where both error types are balanced.
        plt.axvline(
            eer_threshold,
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label="EER threshold",
        )

        plt.xlim(-0.3, 1.05)
        plt.ylim(0, 530)
        plt.xlabel("Cosine similarity")
        plt.ylabel("Number of trials")

        current_axis = plt.gca()

        current_axis.text(
            0.02,
            0.93,
            graph_labels[duration],
            transform=current_axis.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
        )

        current_axis.text(
            eer_threshold + 0.012,
            420,
            f"{eer_threshold:.3f}",
            rotation=90,
            verticalalignment="center",
            horizontalalignment="left",
        )

        current_axis.set_axisbelow(True)

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(
            RESULTS_FOLDER
            / f"score_distributions_{duration}s.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def main():

    """Run the complete experiment."""

    # argparse is an optional way to run only a small subset during testing.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of trials "
            "for a small test run."
        ),
    )

    arguments = parser.parse_args()

   
    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,)


    # The experiment now needs prepared_audio
    if not PREPARED_AUDIO_FOLDER.exists():
        print(
            "Prepared audio folder not found: "
            f"{PREPARED_AUDIO_FOLDER}")

        print("Run prepare_experiment_data.py first.")

        return

    # selected_trials.txt must also have been created by the preparation script.
    if not TRIAL_FILE.exists():
        print(
            f"Trial file not found: "
            f"{TRIAL_FILE}"
        )

        print("Run prepare_experiment_data.py first.")

        return
    
    # The protocol is read first, before checking which audio pairs can actually be used.
    all_trials = read_trials()

    usable_trials = select_usable_trials(all_trials)

    # Limiting trials is useful for checking that the pipeline works before a full run.
    if arguments.limit is not None:
        usable_trials = usable_trials[
            :arguments.limit
        ]

        print(f"Using {len(usable_trials)} trials.")

    if len(usable_trials) == 0:
        print("No usable trials were found.")
        return

    # EER requires both same-speaker and different-speaker trials.
    # Both classes must be present for an equal-error-rate calculation, if not it's not meaningful.
    labels = [
        trial["label"]
        for trial in usable_trials
    ]

    if 0 not in labels or 1 not in labels:
        print(
            "The selected trials must contain "
            "both labels 0 and 1."
        )
        return

    save_selected_trials(usable_trials)

    print("Loading ECAPA-TDNN model...")

    # The pretrained ECAPA-TDNN network converts speech into speaker embeddings.

    model = EncoderClassifier.from_hparams(
        source=(
            "speechbrain/"
            "spkrec-ecapa-voxceleb"
        ),
        savedir=str(MODEL_FOLDER),
        run_opts={"device": "cpu"},
    )

    # Generation of all trial scores.
    results_dataframe = run_experiment(
        model,
        usable_trials,
    )

    # The raw trial scores are reduced to one set of statistics per duration.
    summary_dataframe = create_summary(
        results_dataframe
    )

    create_eer_graph(summary_dataframe)

    create_average_score_graph(
        summary_dataframe
    )

    create_median_score_graph(
        summary_dataframe
    )

    create_score_distribution_graphs(
        results_dataframe
    )

    print(
        f"Results saved in: "
        f"{RESULTS_FOLDER.resolve()}"
    )


if __name__ == "__main__":
    main()