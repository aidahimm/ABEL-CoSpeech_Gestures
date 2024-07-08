import pandas as pd
import os
import csv
import json

predictions_df = pd.read_csv('predicted_joint_positions.csv')

def drop_engineered_features(df):
    col_to_drop = df.filter(regex='^_rel_|^_angular_')
    joint_features = df.drop(columns=[col for col in col_to_drop.columns])
    return joint_features

pure_joint = drop_engineered_features(predictions_df)

empty_json_template = [{
    "Hips": {
        "position": [],
        "rotation": [],
        "offset": [8.70151, 92.36, -22.8466],
        "parent": None
    },
    "Spine": {
        "position": [],
        "rotation": [],
        "offset": [0, 14.0932, -1.51142],
        "parent": "Hips"
    },    
    "Spine1": {
        "position": [],
        "rotation": [],
        "offset": [0, 8.6966, 0],
        "parent": "Spine"
    },
    "Spine2": {
        "position": [],
        "rotation": [],
        "offset": [0, 8.6966, 0],
        "parent": "Spine1"
    },
    "Spine3": {
        "position": [],
        "rotation": [],
        "offset": [0, 8.6966, 0],
        "parent": "Spine2"
    },
    "RightShoulder": {
        "position": [],
        "rotation": [],
        "offset": [0, 7.98668, 5.19711],
        "parent": "Spine3"
    },
    "RightArm": {
        "position": [],
        "rotation": [],
        "offset": [0, 18.3158, 0],
        "parent": "RightShoulder"
    },
    "RightForeArm": {
        "position": [],
        "rotation": [],
        "offset": [0, 28.8742, 0],
        "parent": "RightArm"
    },
    "RightHand": {
        "position": [],
        "rotation": [],
        "offset": [0, 23.8772, 0],
        "parent": "RightForeArm"
    },
    "LeftShoulder": {
        "position": [],
        "rotation": [],
        "offset": [0, 7.98668, 5.19711],
        "parent": "Spine3"
    },
    "LeftArm": {
        "position": [],
        "rotation": [],
        "offset": [0, 18.3158, 0],
        "parent": "LeftShoulder"
    },
    "LeftForeArm": {
        "position": [],
        "rotation": [],
        "offset": [0, 28.8742, 0],
        "parent": "LeftArm"
    },
    "LeftHand": {
        "position": [],
        "rotation": [],
        "offset": [0, 23.8772, 0],
        "parent": "LeftForeArm"
    }
}]

# Function to populate the JSON template
def populate_json_template(template, predictions):
    all_frames = []
    for index, row in predictions.iterrows():
        frame_data = json.loads(json.dumps(template))  # Deep copy of the template for each frame
        for joint in frame_data[0].keys():
            if joint == "Hips":
                frame_data[0][joint]['position'] = [
                    row[f"{joint}_position_x"],
                    row[f"{joint}_position_y"],
                    row[f"{joint}_position_z"]
                ]
                frame_data[0][joint]['rotation'] = [
                    row[f"{joint}_rotation_x"],
                    row[f"{joint}_rotation_y"],
                    row[f"{joint}_rotation_z"]
            ]
            else:
                frame_data[0][joint]['position'] = [0,0,0]
                frame_data[0][joint]['rotation'] = [
                    row[f"{joint}_rotation_x"],
                    row[f"{joint}_rotation_y"],
                    row[f"{joint}_rotation_z"]
                ]
        all_frames.append(frame_data[0])
    return all_frames

# Populate the JSON template with the predictions
populated_json = populate_json_template(empty_json_template, pure_joint)

# Save the populated JSON to a file
with open('populated_joint_positions.json', 'w') as f:
    json.dump(populated_json, f, indent=4)

print("Populated JSON saved as 'populated_joint_positions.json'")