# Speaker Verification and Recording Duration

## About the project

This project looks at whether the length of a speech recording affects speaker verification.

Speaker verification means checking whether two recordings belong to the same person. In this project, I compare recordings with three different lengths:

* 3 seconds
* 5 seconds
* 10 seconds

The first recording in each pair stays the same. The second recording is shortened to one of these three lengths.

A pretrained speaker-recognition model is then used to compare the two recordings.



## Research question

The main question of this project is:

*Does speaker verification work better when the test recording is longer?*

I expect that longer recordings may give the model more information about the speaker’s voice, but the purpose of the experiment is to test this.



## Dataset

The project uses the VoxCeleb1 dataset.

The verification file should be placed here:

data/voxceleb1/veri_test2.txt

The audio files should be placed here:

data/voxceleb1/wav/

The verification file contains pairs of recordings.

A label of `1` means that both recordings belong to the same speaker.

A label of `0` means that the recordings belong to different speakers.

The VoxCeleb audio files are not included in this project and have to be downloaded separately.



## Project files

The main files are:

prepare_experiment_data.py
run_experiment.py
requirements.txt
README.md



### `prepare_experiment_data.py`

This script prepares the audio files for the experiment.

It checks whether the recordings exist and whether the second recording is at least 10 seconds long.

It then creates:

* a 3-second version;
* a 5-second version;
* a 10-second version.

The script also saves a list of the selected trials and creates a CSV file with information about the prepared recordings.



### `run_experiment.py`

This script runs the actual speaker-verification experiment.

It loads the recordings, creates speaker embeddings, compares them, and calculates the results for the three recording lengths.

It also saves the results as CSV files and creates graphs.



## Required packages

The project uses the following Python packages:

torch
torchaudio
speechbrain
soundfile
scipy
scikit-learn
pandas
matplotlib
tqdm

They can be installed using:

pip install -r requirements.txt



## Folder structure

The project folder should look similar to this:

speaker-duration-project/
│
├── prepare_experiment_data.py
├── run_experiment.py
├── requirements.txt
├── README.md
│
├── data/
│   └── voxceleb1/
│       ├── veri_test2.txt
│       └── wav/
│
├── results/
│
└── saved_models/

The `results` and `saved_models` folders are created automatically when needed.




## How to run the project

First, open a terminal and move into the main project folder:

cd /path/to/speaker-duration-project


## Step 1: Install the packages

Run:

pip install -r requirements.txt

Using a virtual environment is recommended, but it is not required.


## Step 2: Add the VoxCeleb files

Place `veri_test2.txt` inside:

data/voxceleb1/

Place the VoxCeleb speaker folders inside:

data/voxceleb1/wav/

For example:

data/voxceleb1/wav/id10270/
data/voxceleb1/wav/id10271/


## Step 3: Prepare the audio

Run:

python prepare_experiment_data.py

This prepares all usable recordings.


The prepared files are saved inside:


data/voxceleb1/prepared_audio/


## Step 4: Run the experiment

Run:

python run_experiment.py

The first run may take longer because the pretrained model may need to be downloaded.



## What the experiment does

For every verification pair, the script:

1. Loads the first recording.
2. Loads the second recording.
3. Shortens the second recording to 3, 5, or 10 seconds.
4. Converts both recordings into speaker embeddings.
5. Compares the embeddings using cosine similarity.
6. Saves the result.

The same verification pairs are used for all three recording lengths.



## Results

The results are saved inside the `results` folder.


### `selected_trials.txt`

This file shows the exact recording pairs used in the experiment.


### `trial_scores.csv`

This file contains the cosine-similarity score for every trial.


### `summary_results.csv`

This file contains the main results for each recording length.

It includes:

* number of trials;
* Equal Error Rate;
* average same-speaker score;
* average different-speaker score;
* median same-speaker score;
* median different-speaker score.



### Graphs

The script creates several graphs:

eer_by_duration.png
average_scores.png
median_scores.png
score_distributions_3s.png
score_distributions_5s.png
score_distributions_10s.png

These graphs make it easier to compare the results for the three recording lengths.



## Equal Error Rate

The main measurement used in this project is Equal Error Rate, also called EER.

EER is a common measurement in speaker verification.

A lower EER means that the system made fewer errors.

The EER values for 3, 5, and 10 seconds can therefore be compared to see whether longer recordings improve performance.



## Cosine similarity

Cosine similarity is used to compare the two speaker embeddings.

A higher score usually means that the two recordings are more similar.

Same-speaker pairs should generally have higher scores than different-speaker pairs.



## Important choices in the experiment

Only the second recording is shortened.

The first recording stays at its original length.

The beginning of the second recording is always used. The script does not choose a random part of the recording.

All audio is changed to mono and resampled to 16 kHz when needed.

These choices keep the experiment consistent.



## Limitations

This project only compares three recording lengths.

It also always uses the beginning of the second recording. A different part of the same recording might produce a different result.

The quality of the recordings is not controlled. Background noise, microphone quality, and speaking style may affect the scores.

The experiment also uses a pretrained model, so the results depend on how well that model works with the VoxCeleb recordings.

Only recordings where the second file is at least 10 seconds long are included. This means that some official verification pairs are left out.



## Possible future work

The project could be extended by:

* testing more recording lengths;
* testing recordings shorter than 3 seconds;
* using random parts of recordings;
* comparing different speaker-verification models;
* adding background noise;
* testing whether the differences are statistically significant.



## Use of AI

The use of AI-assisted tools in this project is documented in ACKNOWLEDGEMENT.md.