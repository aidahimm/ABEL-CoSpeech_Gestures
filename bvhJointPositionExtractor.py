import zipfile
import os
import json
import re

class BVHReader:
    def __init__(self, file_content):
        self.lines = file_content.splitlines()
        self.joints = []
        self.frames = []
        self.joint_offsets = {}
        self.channel_info = {}
        self.joint_hierarchy = {}
        self.parse_bvh()

    def parse_bvh(self):
        joint_stack = []
        current_joint = None
        parsing_hierarchy = True

        for line in self.lines:
            if "MOTION" in line:
                parsing_hierarchy = False
                continue

            tokens = line.split()
            if not tokens:
                continue

            if parsing_hierarchy:
                if tokens[0] in {"ROOT", "JOINT"}:
                    joint_name = tokens[1]
                    if joint_stack:
                        parent_joint = joint_stack[-1]
                        self.joint_hierarchy[joint_name] = parent_joint
                    else:
                        self.joint_hierarchy[joint_name] = None
                    joint_stack.append(joint_name)
                    self.joints.append(joint_name)
                    current_joint = joint_name
                elif tokens[0] == "End":
                    joint_stack.append(current_joint + "_end")
                    self.joints.append(current_joint + "_end")
                elif tokens[0] == "{":
                    continue
                elif tokens[0] == "}":
                    joint_stack.pop()
                elif tokens[0] == "OFFSET":
                    offset = list(map(float, tokens[1:]))
                    self.joint_offsets[current_joint] = offset
                elif tokens[0] == "CHANNELS":
                    num_channels = int(tokens[1])
                    channels = tokens[2:2 + num_channels]
                    self.channel_info[current_joint] = channels
            else:
                if tokens[0] == "Frames:":
                    num_frames = int(tokens[1])
                elif tokens[0] == "Frame" and tokens[1] == "Time:":
                    frame_time = float(tokens[2])
                else:
                    frame_data = list(map(float, tokens))
                    frame_dict = {}
                    data_index = 0
                    for joint in self.joints:
                        if joint in self.channel_info:
                            joint_channels = self.channel_info[joint]
                            joint_data = frame_data[data_index:data_index + len(joint_channels)]
                            frame_dict[joint] = {
                                "channels": joint_channels,
                                "data": joint_data
                            }
                            data_index += len(joint_channels)
                    self.frames.append(frame_dict)

    def get_joint_positions(self, joints_to_include):
        joint_positions = []
        for frame in self.frames:
            frame_positions = {}
            for joint in joints_to_include:
                if joint in frame:
                    position_data = frame[joint]["data"]
                    joint_channels = frame[joint]["channels"]
                    joint_offset = self.joint_offsets.get(joint, [0, 0, 0])
                    
                    # Split the data into positions and rotations based on channels
                    position = [0, 0, 0]
                    rotation = [0, 0, 0]
                    for i, channel in enumerate(joint_channels):
                        if 'position' in channel.lower():
                            if 'x' in channel.lower():
                                position[0] = position_data[i]
                            elif 'y' in channel.lower():
                                position[1] = position_data[i]
                            elif 'z' in channel.lower():
                                position[2] = position_data[i]
                        elif 'rotation' in channel.lower():
                            if 'x' in channel.lower():
                                rotation[0] = position_data[i]
                            elif 'y' in channel.lower():
                                rotation[1] = position_data[i]
                            elif 'z' in channel.lower():
                                rotation[2] = position_data[i]

                    frame_positions[joint] = {
                        "position": position,
                        "rotation": rotation,
                        "offset": joint_offset,
                        "parent": self.joint_hierarchy.get(joint)
                    }
            joint_positions.append(frame_positions)
        return joint_positions


def process_bvh_files_from_zip(zip_path, output_dir, joints_to_include):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            #VSCODE KEEPS CRASHING
            if file_name.endswith('.bvh'):
                print(f"Processing file: {file_name}")
                with zip_ref.open(file_name) as file:
                    bvh_data = file.read().decode('utf-8')
                    reader = BVHReader(bvh_data)
                    joint_positions = reader.get_joint_positions(joints_to_include)

                    # Define the output JSON file path
                    json_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.json")

                    # Save the joint positions to the JSON file
                    with open(json_output_path, 'w') as json_file:
                        json.dump(joint_positions, json_file, indent=4)

                    print(f"Joint positions saved to: {json_output_path}")

# Input directory containing BVH files
bvh_zip = '/Volumes/NO NAME/Thesis/AllBvhFromClap.zip'

# Output directory to save JSON files
output_dir = '/Volumes/NO NAME/ABEL-body-motion/JointPositions'

# List of joints to include
joints_to_include = ['Hips','Spine', 'Spine1','Spine2', 'Spine3','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']

# Process the BVH files and save the extracted positions
process_bvh_files_from_zip(bvh_zip, output_dir, joints_to_include)