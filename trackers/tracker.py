from ultralytics import YOLO  # type: ignore
import supervision as sv
import numpy as np
from typing import List
import pickle
import pandas as pd
import os
import cv2
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_bbox, get_width_bbox, get_foot_position


class Tracker:
    def __init__(self, model_path:str) -> None:
        # Initialize YOLO 
        self.model = YOLO(model_path)
        # Initialize ByteTrack tracker from supervision library
        self.tracker = sv.ByteTrack()
    
    def add_pos_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info.get('bbox', [])
                    if object == 'football':
                        position = get_center_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

                

    def interpolate_ball_pos(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox', []) for x in ball_positions]

        df_ball_pos = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_pos = df_ball_pos.interpolate()
        df_ball_pos = df_ball_pos.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_pos.to_numpy().tolist()]

        return ball_positions
        
    def detect_frames(self, frames: List[np.ndarray]):
        batch_size = 8
        detections = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            # Predict detections for the current batch
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.3)
            detections.extend(detections_batch)
        
        return detections
    
    def get_object_tracks(self, frames: List[np.ndarray], read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Detect objects in the frames
        detections = self.detect_frames(frames)

        tracks = {
            "player": [],  # Corrected from "playerss"
            "football": [],
        }

        for frame_num, detection in enumerate(detections):
            # Get class names and map them to their IDs
            cls_names = detection.names
            cls_names_inverted = {v: k for k, v in cls_names.items()}

            # Convert YOLO detections to supervision Detections
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["player"].append({})
            tracks["football"].append({})

            for frame_d in detections_with_tracks:
                print(frame_d)
                bbox = frame_d[0].tolist()
                class_info = frame_d[5]  
                cls_id = class_info['class_name']
                track_id = frame_d[4]
                
                if cls_id == 'players':
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == 'football':
                    tracks["football"][frame_num][track_id] = {"bbox": bbox}
                else:
                    print(f"Unknown class ID: {cls_id}")

        if stub_path is not None:
            # Save the tracks to a file
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_circle(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_bbox(bbox)
        width = get_width_bbox(bbox)

        # Draw the ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(.36 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=240,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Define the rectangle where the track_id (player_counter or football_counter) will be displayed
        rect_width = 80
        rect_height = 20
        x1_rect = x_center - rect_width // 2
        x2_rect = x_center + rect_width // 2
        y1_rect = (y2 - rect_height // 2) + 20
        y2_rect = (y2 + rect_height // 2) + 20

        if track_id is not None:
            # Draw the rectangle
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            
            text = str(track_id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x_center - text_size[0] // 2  # Center the text horizontally
            text_y = y1_rect + (rect_height + text_size[1]) // 2  # Center the text vertically
            
            # Draw the text on the rectangle
            cv2.putText(frame, text, (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame
            
        
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[3])
        x,_ = get_center_bbox(bbox)



        triangle_points = np.array([
            [x, y],
            [x - 10, y + 20],
            [x + 10, y + 20]
        ])
        cv2.drawCountours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)
        return frame

    def draw_annotations(self, frame_v, tracks):
        output_video_frame = []
        for frame_num, frame in enumerate(frame_v):
            frame_copy = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["football"][frame_num]

            # Draw players 
            for track_id, player in player_dict.items():
                frame_copy = self.draw_circle(frame_copy, player["bbox"], (0, 0, 255), track_id)

            # Draw footballs 
            for track_id, football in ball_dict.items():
                frame_copy = self.draw_triangle(frame_copy, football["bbox"], (0, 255, 0))

            output_video_frame.append(frame_copy)

        return output_video_frame