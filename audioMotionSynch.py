import os
import json
import re
import csv

# Function to load alignment times from CSV file
def load_alignment_times(filepath):
    alignment_data = {}
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract the number from the 'file' column
            number = re.search(r'\d+', row['Take']).group()
            alignment_data[number] = {
                'MotionStart': float(row['MotionStart']),
                'AudioStart': float(row['AudioStart'])
            }
    return alignment_data

# Directory containing Joint Position JSON files
jp_json_dir = '/Volumes/NO NAME/ABEL-body-motion/JointPositions'

# List all Joint Position JSON files in the directory
jp_json_files = [f for f in os.listdir(jp_json_dir) if f.endswith('.json')]

# Directory containing Transcript JSON files
tr_json_dir = '/Volumes/NO NAME/ABEL-body-motion/Transcripts'

# List all Joint Position JSON files in the directory
tr_json_files = [f for f in os.listdir(tr_json_dir) if f.endswith('.json')]

# Sorting the files by numbering
jp_json_files.sort()
tr_json_files.sort()

# Function to create pairs of JointPositions and Transcript files
def list_and_pair_files():
    pairs = []
    for joint_file, transcript_file in zip(jp_json_files, tr_json_files):
        joint_number = re.search(r'\d+', joint_file).group()
        transcript_number = re.search(r'\d+', transcript_file).group()
        if joint_number == transcript_number:
            pairs.append((joint_file, transcript_file, joint_number))
    return pairs

def load_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Function to extract joint positions from JSON data
def extract_joint_positions(json_data):
    joint_positions = []
    for frame in json_data:
        frame_data = {}
        for joint, values in frame.items():
            if joint not in ['partial', 'result', 'text']:
                frame_data[joint] = {
                    'position': values.get('position', []),
                    'rotation': values.get('rotation', []),
                    'offset': values.get('offset', []),
                    'parent': values.get('parent')
                }
        if frame_data:
            joint_positions.append(frame_data)
    return joint_positions

import numpy as np
import json

def extract_frames_from_bvh(bvh_file):
    # Function to parse BVH file and extract frames
    # This is a simplified placeholder function
    frames = []
    with open(bvh_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Frame" in line:
                frame_data = json.loads(line)  # Assuming frame data is in JSON format
                frames.append(frame_data)
    return frames

# def create_empty_frame_template():
#     return {
#         "Hips": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": None},
#         "Spine": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Hips"},
#         "Spine1": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Spine"},
#         "Spine2": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Spine1"},
#         "Spine3": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Spine2"},
#         "RightShoulder": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Spine3"},
#         "RightArm": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "RightShoulder"},
#         "RightForeArm": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "RightArm"},
#         "RightHand": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "RightForeArm"},
#         "LeftShoulder": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "Spine3"},
#         "LeftArm": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "LeftShoulder"},
#         "LeftForeArm": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "LeftArm"},
#         "LeftHand": {"position": [0, 0, 0], "rotation": [0, 0, 0], "offset": [0, 0, 0], "parent": "LeftForeArm"}
#     }

# Function to extract transcript from JSON data
def extract_transcript(json_data):
    transcript = []
    for entry in json_data:
        if 'result' in entry:
            for word in entry['result']:
                transcript.append({
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word['conf']
                })
    return transcript

# Function to adjust timestamps using alignment times
def adjust_timestamps(alignment_data, file_number, frame_time):
    joint_start_sec = alignment_data[file_number]['MotionStart']
    audio_start_sec = alignment_data[file_number]['AudioStart']
    joint_start_frame = joint_start_sec / frame_time
    return joint_start_frame, joint_start_sec, audio_start_sec

# Function to align joint positions with transcript using adjusted timestamps
def align_data(joint_positions, transcript, joint_start_frame, joint_start_sec, audio_start_sec, frame_time):
    aligned_data = []
    transcript_index = 0

    for word_data in transcript:
        start_time = word_data['start']
        if start_time < audio_start_sec:
            transcript_index +=1
        else:
            break

    word_data = transcript[transcript_index]
    start_time = word_data['start']
    end_time = word_data['end'] 
    
    for i, frame in enumerate(joint_positions):
        frame_time_adjusted = (i + joint_start_frame) * frame_time# Timing of the joint position using the provided frame_time in the bvh files
        if transcript_index < len(transcript):
            word_data = transcript[transcript_index]
            start_time = word_data['start']
            end_time = word_data['end'] 

            if start_time <= frame_time_adjusted <= end_time:
                aligned_frame = {
                    'frame': i,
                    'time': frame_time_adjusted,
                    'joint_positions': frame,
                    'transcript': word_data['word']
                }
                aligned_data.append(aligned_frame)

            # If the joint position timing 
            elif frame_time_adjusted > end_time:
                transcript_index += 1

            else:
                aligned_frame = {
                    'frame': i,
                    'time': frame_time_adjusted,
                    'joint_positions': frame,
                    'transcript': '000'
                }
                aligned_data.append(aligned_frame)
        else:
            aligned_frame = {
                'frame': i,
                'time': frame_time_adjusted,
                'joint_positions': frame,
                'transcript': '999'
            }
            aligned_data.append(aligned_frame)

    return aligned_data

# Function to process all pairs and save aligned data
def process_all_pairs(pairs, jp_json_dir, tr_json_dir, output_dir, frame_time):
        for joint_file, transcript_file, file_number  in pairs:
            # Load JSON data
            joint_data = load_json_file(os.path.join(jp_json_dir, joint_file))
            transcript_data = load_json_file(os.path.join(tr_json_dir, transcript_file))
            
            # Extract and align data
            joint_positions = extract_joint_positions(joint_data)
            transcript = extract_transcript(transcript_data)
            
            # Adjust timestamps using alignment data
            joint_start_frame, joint_start_sec, audio_start_sec = adjust_timestamps(alignment_data, file_number, frame_time)
            aligned_data = align_data(joint_positions, transcript, joint_start_frame, joint_start_sec, audio_start_sec, frame_time)        
            
            # Define output filepath
            output_filename = f"aligned_{os.path.basename(joint_file)}"
            output_filepath = os.path.join(output_dir, output_filename)
            
            # Save aligned data
            with open(output_filepath, 'w') as file:
                json.dump(aligned_data, file, indent=4)
            
            print(f"Aligned data saved to: {output_filepath}")


output_dir = '/Volumes/NO NAME/ABEL-body-motion/AllignedSpeechMotion'
os.makedirs(output_dir, exist_ok=True)
alignment_file = '/Volumes/NO NAME/ABEL-body-motion/AlignmentTimes.csv'
alignment_data = load_alignment_times(alignment_file)
frame_time = 0.00833333
pairs = list_and_pair_files()
process_all_pairs(pairs, jp_json_dir, tr_json_dir, output_dir, frame_time)