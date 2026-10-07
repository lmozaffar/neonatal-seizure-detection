import mne
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(data_dir="data/raw"):
    # Implementation for loading data
    ##Returns mne raw object containing the loaded data and path to EDF file that was loaded
    
    data_dir = Path(data_dir)

    #Find EDF files
    edf_files = list(data_dir.glob("*.edf"))

    #Return error if no EDF files found
    if len(edf_files) == 0:
        raise ValueError("No EDF files found in the specified directory.")

    print("Number of EDF Files:", len(edf_files))

    #Load first patient
    edf_path = edf_files[0]
    raw = mne.io.read_raw_edf(edf_path, preload=False,
        verbose=False)

    return raw, edf_files

def inspect_data(raw, edf_path):
    #Prints info about one patient

    sampling_freq = raw.info['sfreq']
    number_channels = len(raw.ch_names)
    number_samples = raw.n_times

    print(f"Sampling Frequency: {sampling_freq} Hz")
    print(f"Number of Channels: {number_channels}")
    print(f"Number of Samples: {number_samples}")

    duration_seconds = number_samples / sampling_freq
    duration_minutes = duration_seconds / 60

    print(f"Duration: {duration_seconds} seconds ({duration_minutes} minutes)")

    print("\nChannels:")

    for i, channel in enumerate(raw.ch_names):
        print(f"{i}: {channel}")


def main():
    # Load one Helsinki EEG recording
    raw, edf_path = load_data()

    # Make sure everything looks correct
    inspect_data(raw, edf_path)

if __name__ == "__main__":
    main()
