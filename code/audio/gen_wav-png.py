import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_waveform_png(wav_file, output_png):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)

    # Create the plot without labels, titles, or axes
    plt.figure(figsize=(4, 4))
    plt.axis('off')  # Remove axes
    librosa.display.waveshow(y, sr=sr, alpha=0.8)

    # Save the figure as a PNG file
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# Example usage
wav_filename = "../MVTEC-AD-WAV/train/transistor_train_good_000.wav"  # Replace with your actual .wav file
png_filename = "transistor.png"  # Output PNG filename
generate_waveform_png(wav_filename, png_filename)


