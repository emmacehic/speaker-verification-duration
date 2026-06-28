"""
run_experiment.py

This program tests whether the length of a test recording affects
speaker-verification performance.

I compare three test-recording lengths:

- 3 seconds
- 5 seconds
- 10 seconds

For each verification pair:

- The first recording stays unchanged.
- The second recording is shortened.
- The two recordings are converted into speaker embeddings.
- The embeddings are compared using cosine similarity.

The program also calculates Equal Error Rate (EER).
A lower EER means better speaker-verification performance.
"""

from pathlib import Path
import argparse

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



# Project folders and experiment settings


AUDIO_FOLDER = Path("data/voxceleb1/wav")
TRIAL_FILE = Path("data/voxceleb1/veri_test2.txt")
RESULTS_FOLDER = Path("results")
MODEL_FOLDER = Path("saved_models/ecapa_tdnn")

SAMPLE_RATE = 16000
TEST_DURATIONS = [3, 5, 10]



# Read the VoxCeleb verification-pair file


def read_trials():
    """
    Read all verification pairs from veri_test2.txt.

    Each line should have this format:

    1 first_recording.wav second_recording.wav
    0 first_recording.wav second_recording.wav

    1 means the recordings are from the same speaker.
    0 means the recordings are from different speakers.
    """

    trials = []

    with TRIAL_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                print(f"Skipping line {line_number}: wrong format.")
                continue

            label_text, first_file, second_file = parts

            try:
                label = int(label_text)
            except ValueError:
                print(f"Skipping line {line_number}: invalid label.")
                continue

            if label not in [0, 1]:
                print(f"Skipping line {line_number}: label must be 0 or 1.")
                continue

            trial = {
                "label": label,
                "first_file": first_file,
                "second_file": second_file,
                "first_path": AUDIO_FOLDER / first_file,
                "second_path": AUDIO_FOLDER / second_file,
            }

            trials.append(trial)

    return trials



# Find the duration of an audio recording


def get_audio_duration(audio_path):
    """
    Return the duration of an audio file in seconds.
    """

    audio_info = sf.info(audio_path)

    duration = audio_info.frames / audio_info.samplerate

    return duration



# Select the trials that can be used


def select_usable_trials(all_trials):
    """
    Keep only trials where:

    - Both audio files exist.
    - The second recording is at least 10 seconds long.
    """

    usable_trials = []

    missing_files = 0
    unreadable_files = 0
    short_files = 0

    for trial in tqdm(all_trials, desc="Checking trial files"):

        first_path = trial["first_path"]
        second_path = trial["second_path"]

        if not first_path.exists() or not second_path.exists():
            missing_files += 1
            continue

        try:
            second_duration = get_audio_duration(second_path)
        except Exception:
            unreadable_files += 1
            continue

        if second_duration < 10:
            short_files += 1
            continue

        usable_trials.append(trial)

    print()
    print("Trial check")
    print("-" * 40)
    print(f"Original trials:       {len(all_trials)}")
    print(f"Usable trials:         {len(usable_trials)}")
    print(f"Missing files:         {missing_files}")
    print(f"Unreadable files:      {unreadable_files}")
    print(f"Files under 10 sec:    {short_files}")
    print("-" * 40)

    return usable_trials



# Load an audio file


def load_audio(audio_path):
    """
    Load an audio file and return it as a PyTorch tensor.

    The audio is also:

    - changed to mono
    - resampled to 16 kHz when needed
    """

    audio, original_sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    audio = audio.mean(axis=1)

    if original_sample_rate != SAMPLE_RATE:

        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            original_sample_rate,
        )

        audio = audio.astype("float32")

    waveform = torch.tensor(audio, dtype=torch.float32)

    return waveform



# Shorten the test recording


def crop_audio(waveform, duration_seconds):
    """
    Keep only the first part of an audio recording.
    """

    number_of_samples = duration_seconds * SAMPLE_RATE

    cropped_waveform = waveform[:number_of_samples]

    return cropped_waveform



# Create a speaker embedding


def extract_embedding(model, waveform):
    """
    Use the ECAPA-TDNN model to create a speaker embedding.
    """

    batch = waveform.unsqueeze(0)

    with torch.inference_mode():

        embedding = model.encode_batch(
            batch,
            normalize=True,
        )

    embedding = embedding.squeeze()

    embedding = F.normalize(embedding, dim=0)

    return embedding.cpu()



# Compare two speaker embeddings


def calculate_cosine_similarity(first_embedding, second_embedding):
    """
    Calculate cosine similarity between two embeddings.
    """

    score = F.cosine_similarity(
        first_embedding.unsqueeze(0),
        second_embedding.unsqueeze(0),
    )

    return float(score.item())



# Calculate Equal Error Rate


def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate.

    EER is the point where the false-acceptance rate and
    false-rejection rate are approximately equal.
    """

    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    false_negative_rates = 1 - true_positive_rates

    differences = np.abs(
        false_positive_rates - false_negative_rates
    )

    closest_index = np.argmin(differences)

    eer = (
        false_positive_rates[closest_index]
        + false_negative_rates[closest_index]
    ) / 2

    threshold = thresholds[closest_index]

    return float(eer), float(threshold)



# Save the trial pairs used in the experiment


def save_selected_trials(trials):
    """
    Save the exact verification pairs used in the experiment.
    """

    output_file = RESULTS_FOLDER / "selected_trials.txt"

    with output_file.open("w", encoding="utf-8") as file:

        for trial in trials:

            file.write(
                f"{trial['label']} "
                f"{trial['first_file']} "
                f"{trial['second_file']}\n"
            )

    print(f"Selected trials saved to {output_file}")



# Run the three duration conditions


def run_experiment(model, trials):
    """
    Run the experiment for 3, 5, and 10 seconds.
    """

    all_results = []

    first_embedding_cache = {}
    test_audio_cache = {}

    for duration in TEST_DURATIONS:

        print()
        print("=" * 50)
        print(f"Running the {duration}-second condition")
        print("=" * 50)

        for trial_number, trial in enumerate(
            tqdm(trials, desc=f"{duration}-second trials"),
            start=1,
        ):

            first_path = trial["first_path"]
            second_path = trial["second_path"]

            first_key = str(first_path)
            second_key = str(second_path)

            if first_key not in first_embedding_cache:

                first_waveform = load_audio(first_path)

                first_embedding = extract_embedding(
                    model,
                    first_waveform,
                )

                first_embedding_cache[first_key] = first_embedding

            first_embedding = first_embedding_cache[first_key]

            if second_key not in test_audio_cache:

                second_waveform = load_audio(second_path)

                test_audio_cache[second_key] = second_waveform

            second_waveform = test_audio_cache[second_key]

            cropped_waveform = crop_audio(
                second_waveform,
                duration,
            )

            second_embedding = extract_embedding(
                model,
                cropped_waveform,
            )

            similarity_score = calculate_cosine_similarity(
                first_embedding,
                second_embedding,
            )

            result = {
                "trial_number": trial_number,
                "duration_seconds": duration,
                "label": trial["label"],
                "first_file": trial["first_file"],
                "second_file": trial["second_file"],
                "cosine_similarity": similarity_score,
            }

            all_results.append(result)

    results_dataframe = pd.DataFrame(all_results)

    output_file = RESULTS_FOLDER / "trial_scores.csv"

    results_dataframe.to_csv(output_file, index=False)

    print()
    print(f"Individual scores saved to {output_file}")

    return results_dataframe



# Create a summary of the results


def create_summary(results_dataframe):
    """
    Calculate EER, mean scores, and median scores
    for each test duration.
    """

    summary_rows = []

    for duration in TEST_DURATIONS:

        duration_results = results_dataframe[
            results_dataframe["duration_seconds"] == duration
        ]

        labels = duration_results["label"].to_numpy()

        scores = duration_results[
            "cosine_similarity"
        ].to_numpy()

        eer, threshold = calculate_eer(labels, scores)

        same_speaker_results = duration_results[
            duration_results["label"] == 1
        ]

        different_speaker_results = duration_results[
            duration_results["label"] == 0
        ]

        mean_same_score = same_speaker_results[
            "cosine_similarity"
        ].mean()

        mean_different_score = different_speaker_results[
            "cosine_similarity"
        ].mean()

        median_same_score = same_speaker_results[
            "cosine_similarity"
        ].median()

        median_different_score = different_speaker_results[
            "cosine_similarity"
        ].median()

        mean_score_difference = (
            mean_same_score - mean_different_score
        )

        median_score_difference = (
            median_same_score - median_different_score
        )

        summary_row = {
            "duration_seconds": duration,
            "number_of_trials": len(duration_results),
            "same_speaker_trials": len(same_speaker_results),
            "different_speaker_trials": len(
                different_speaker_results
            ),
            "eer": eer,
            "eer_percent": eer * 100,
            "eer_threshold": threshold,
            "mean_same_speaker_score": mean_same_score,
            "mean_different_speaker_score": mean_different_score,
            "mean_score_difference": mean_score_difference,
            "median_same_speaker_score": median_same_score,
            "median_different_speaker_score": median_different_score,
            "median_score_difference": median_score_difference,
        }

        summary_rows.append(summary_row)

    summary_dataframe = pd.DataFrame(summary_rows)

    output_file = RESULTS_FOLDER / "summary_results.csv"

    summary_dataframe.to_csv(output_file, index=False)

    print()
    print("Summary results")
    print("=" * 120)
    print(summary_dataframe.to_string(index=False))
    print("=" * 120)

    print()
    print(f"Summary saved to {output_file}")

    return summary_dataframe



# Create the EER graph


def create_eer_graph(summary_dataframe):
    """
    Create a line graph showing the EER for each duration.
    """

    plt.figure(figsize=(7, 5))

    plt.plot(
        summary_dataframe["duration_seconds"],
        summary_dataframe["eer_percent"],
        marker="o",
    )

    plt.xlabel("Test-recording duration in seconds")
    plt.ylabel("Equal Error Rate (%)")
    plt.title("Effect of Test Duration on Speaker Verification")
    plt.xticks(TEST_DURATIONS)
    plt.grid(True)
    plt.tight_layout()

    output_file = RESULTS_FOLDER / "eer_by_duration.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"EER graph saved to {output_file}")



# Create the average-score graph


def create_average_score_graph(summary_dataframe):
    """
    Create a bar graph comparing average same-speaker and
    different-speaker similarity scores.
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
    plt.ylabel("Average cosine similarity")
    plt.title("Average Speaker-Verification Scores")

    plt.xticks(
        positions,
        [f"{duration} seconds" for duration in durations],
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

    output_file = RESULTS_FOLDER / "average_scores.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Average-score graph saved to {output_file}")



# Create the median-score graph


def create_median_score_graph(summary_dataframe):
    """
    Create a bar graph comparing median same-speaker and
    different-speaker similarity scores.
    """

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
        [f"{duration} seconds" for duration in durations],
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

    output_file = RESULTS_FOLDER / "median_scores.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Median-score graph saved to {output_file}")



# Create score-distribution graphs

def create_score_distribution_graphs(results_dataframe):
    """
    Create one separate score-distribution graph
    for each recording duration.

    Each graph shows the EER threshold.
    """

    # Labels shown inside the three graphs.
    graph_labels = {
        3: "(a) 3-second condition",
        5: "(b) 5-second condition",
        10: "(c) 10-second condition",
    }

    # Using the same bins for all three conditions.
    # This makes the graphs directly comparable.
    histogram_bins = np.linspace(-0.3, 1.05, 41)

    for duration in TEST_DURATIONS:

        duration_results = results_dataframe[
            results_dataframe["duration_seconds"] == duration
        ]

        same_scores = duration_results[
            duration_results["label"] == 1
        ]["cosine_similarity"]

        different_scores = duration_results[
            duration_results["label"] == 0
        ]["cosine_similarity"]

        # Calculate the EER threshold for this duration.
        labels = duration_results[
            "label"
        ].to_numpy()

        scores = duration_results[
            "cosine_similarity"
        ].to_numpy()

        eer, eer_threshold = calculate_eer(
            labels,
            scores,
        )

        # Create one separate graph.
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

        # Draw the EER threshold.
        plt.axvline(
            eer_threshold,
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label="EER threshold",
        )

        # Keep the same axes for all three graphs.
        plt.xlim(-0.3, 1.05)
        plt.ylim(0, 530)

        plt.xlabel("Cosine similarity")
        plt.ylabel("Number of trials")

        current_axis = plt.gca()

        # Place the graph label inside the upper-left corner.
        current_axis.text(
            0.02,
            0.93,
            graph_labels[duration],
            transform=current_axis.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
        )

        # Place the threshold value vertically beside the line.
        current_axis.text(
            eer_threshold + 0.012,
            420,
            f"{eer_threshold:.3f}",
            rotation=90,
            verticalalignment="center",
            horizontalalignment="left",
        )

        # Add light grid lines behind the histograms.
        current_axis.set_axisbelow(True)

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            loc="upper right",
        )

        plt.tight_layout()

        output_file = (
            RESULTS_FOLDER
            / f"score_distributions_{duration}s.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Distribution graph saved to {output_file}"
        )


# Main part of the program


def main():
    """
    Start the experiment.

    A small test can be run using:

    python run_experiment.py --limit 20
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of trials for a small test run.",
    )

    arguments = parser.parse_args()

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    if not AUDIO_FOLDER.exists():
        print("Audio folder not found:")
        print(AUDIO_FOLDER)
        print()
        print("Add the VoxCeleb1 WAV files before running the program.")
        return

    if not TRIAL_FILE.exists():
        print("Trial file not found:")
        print(TRIAL_FILE)
        print()
        print("Add veri_test2.txt inside data/voxceleb1.")
        return

    print("Reading the verification trial file...")

    all_trials = read_trials()

    print(f"Trial lines found: {len(all_trials)}")

    usable_trials = select_usable_trials(all_trials)

    if arguments.limit is not None:

        usable_trials = usable_trials[:arguments.limit]

        print()
        print(f"Using only {len(usable_trials)} trials for this test.")

    if len(usable_trials) == 0:
        print()
        print("No usable trials were found.")
        return

    labels = [trial["label"] for trial in usable_trials]

    if 0 not in labels or 1 not in labels:
        print()
        print("The selected trials do not contain both labels.")
        print("Use a larger --limit value.")
        return

    save_selected_trials(usable_trials)

    print()
    print("Loading the pretrained ECAPA-TDNN model...")
    print("The first download may take a few minutes.")

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_FOLDER),
        run_opts={"device": "cpu"},
    )

    print("Model loaded successfully.")

    results_dataframe = run_experiment(
        model,
        usable_trials,
    )

    summary_dataframe = create_summary(
        results_dataframe
    )

    create_eer_graph(summary_dataframe)

    create_average_score_graph(summary_dataframe)

    create_median_score_graph(summary_dataframe)

    create_score_distribution_graphs(
        results_dataframe
    )

    print()
    print("=" * 50)
    print("Experiment completed successfully.")
    print("=" * 50)
    print()
    print("The results are saved in:")
    print(RESULTS_FOLDER.resolve())


if __name__ == "__main__":
    main()