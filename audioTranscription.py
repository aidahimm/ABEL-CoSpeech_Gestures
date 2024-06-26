import wave
import json
import zipfile
import os
import io
from vosk import Model, KaldiRecognizer

model = Model("/Volumes/NO NAME/Thesis/vosk-model-en-us-0.22")

def transcribe_audio(audio_file):
    rec = KaldiRecognizer(model, audio_file.getframerate())
    rec.SetWords(True)
    results = []

    while True:
        data = audio_file.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
        else:
            results.append(json.loads(rec.PartialResult()))

    results.append(json.loads(rec.FinalResult()))
    return results


def process_zip_file(zip_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all the files in the zip archive
        file_list = zip_ref.namelist()
        
        for file_name in file_list:
            if file_name.endswith('.wav'):
                print(f"Processing file: {file_name}")
                with zip_ref.open(file_name) as file:
                    # Read the WAV file from the zip archive
                    with wave.open(io.BytesIO(file.read()), 'rb') as wav_file:
                        # Transcribe audio
                        transcript = transcribe_audio(wav_file)

                        # Define the output JSON file path
                        json_output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.json")

                        # Save the transcript to the JSON file
                        with open(json_output_path, 'w') as json_file:
                            json.dump(transcript, json_file, indent=4)

                        print(f"Transcript saved to: {json_output_path}")

# Path to the ZIP file containing WAV files
zip_path = '/Volumes/NO NAME/Thesis/AllAudio.zip'
output_folder = "/Volumes/NO NAME/ABEL-body-motion/Transcripts"
# Process the ZIP file
process_zip_file(zip_path, output_folder)