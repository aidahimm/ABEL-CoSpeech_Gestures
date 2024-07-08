import json

class BVHWriter:
    def __init__(self, json_data):
        self.json_data = json_data

    def write_bvh(self, output_path):
        bvh_content = self.construct_hierarchy()
        bvh_content += self.construct_motion()
        with open(output_path, 'w') as file:
            file.write(bvh_content)
        print(f"BVH file saved to: {output_path}")

    def construct_hierarchy(self):
        bvh_hierarchy = "HIERARCHY\n"
        joint_stack = []
        joint_data = self.json_data[0]  # Assuming the first frame contains the hierarchy

        for joint, details in joint_data.items():
            if details['parent'] is None:
                bvh_hierarchy += self.joint_to_bvh(joint, details, joint_stack)

        return bvh_hierarchy

    def joint_to_bvh(self, joint, details, joint_stack):
        bvh_joint = ""
        indent = "  " * len(joint_stack)
        if len(joint_stack) == 0:
            bvh_joint += f"{indent}ROOT {joint}\n"
        else:
            bvh_joint += f"{indent}JOINT {joint}\n"
        bvh_joint += f"{indent}{{\n"
        joint_stack.append(joint)
        bvh_joint += f"{indent}  OFFSET {details['offset'][0]} {details['offset'][1]} {details['offset'][2]}\n"
        if len(joint_stack) == 1:
            bvh_joint += f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        else:
            bvh_joint += f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n"

        for child_joint, child_details in self.json_data[0].items():
            if child_details['parent'] == joint:
                bvh_joint += self.joint_to_bvh(child_joint, child_details, joint_stack)

        bvh_joint += f"{indent}}}\n"
        joint_stack.pop()
        return bvh_joint

    def construct_motion(self):
        motion_data = "MOTION\n"
        num_frames = len(self.json_data)
        frame_time = 0.00833333  # Set your frame time here
        motion_data += f"Frames: {num_frames}\n"
        motion_data += f"Frame Time: {frame_time}\n"

        for frame in self.json_data:
            frame_line = []
            for joint, details in frame.items():
                frame_line.extend(details['position'])
                frame_line.extend(details['rotation'])
            motion_data += ' '.join(map(str, frame_line)) + '\n'

        return motion_data

# Load the populated JSON
with open('populated_joint_positions.json', 'r') as f:
    populated_json = json.load(f)

# Create a BVHWriter instance
bvh_writer = BVHWriter(populated_json)

# Define the output BVH file path
output_bvh_path = 'rebuilt_prediction_BVH.bvh'

# Write the BVH file
bvh_writer.write_bvh(output_bvh_path)
