import os
import re
import json
import numpy as np
import pandas as pd

# Function to load JSON file
def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Function to extract numeric document ID from filename
def extract_take_id(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric ID found in filename: {filename}")

# Directory containing JSON files
json_dir = '/Volumes/NO NAME/ABEL-body-motion/AllignedSpeechMotion'

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

# Extract document IDs
document_ids = [extract_take_id(f) for f in json_files]

# Function to load JSON file
def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Function to extract features from aligned data with document ID
def extract_features(aligned_data, doc_id):
    features = []
    for frame_data in aligned_data:
        frame = frame_data['frame']
        time = frame_data['time']
        joint_positions = frame_data['joint_positions']
        words = frame_data['transcript']

        # Flatten joint positions, rotations, and offsets
        flat_positions = {}
        flat_parents = {}

        for joint, details in joint_positions.items():
            for axis, pos in zip(['x', 'y', 'z'], details['position']):
                flat_positions[f"{joint}_position_{axis}"] = pos
            for axis, rot in zip(['x', 'y', 'z'], details['rotation']):
                flat_positions[f"{joint}_rotation_{axis}"] = rot
            for axis, off in zip(['x', 'y', 'z'], details['offset']):
                flat_positions[f"{joint}_offset_{axis}"] = off
            flat_parents[f"{joint}_parent"] = details['parent']


        feature = {
            'take_id': doc_id,
            'frame': frame,
            'time': time,
            'word': words,
            **flat_positions,
            **flat_parents,
        }
        features.append(feature)

    return features


# Loop over each JSON file and extract features
for json_file in json_files:
    take_id = extract_take_id(json_file)
    filepath = os.path.join(json_dir, json_file)
    aligned_data = load_json(filepath)
    print(f"Extracting data from take: ", take_id)
    features = extract_features(aligned_data, take_id)
    # Convert features to DataFrame
    df_features = pd.DataFrame(features)
    
    # Save features to CSV
    output_csv = f"/Volumes/NO NAME/ABEL-body-motion/ExtractedFeatures/features_{take_id}.csv"
    df_features.to_csv(output_csv, index=False)

    print(f"Processed and saved features for {json_file} to {output_csv}")

