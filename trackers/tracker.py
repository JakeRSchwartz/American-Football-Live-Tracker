from ultralytics import YOLO  # type: ignore
import supervision as sv
import numpy as np
from typing import List
import pickle
import os
import cv2

class Tracker:
    def __init__(self, model_path:str) -> None:
        # Initialize YOLO 
        self.model = YOLO(model_path)
        # Initialize ByteTrack tracker from supervision library
        self.tracker = sv.ByteTrack()
        pass
        
    def detect_frames(self, frames: List[np.ndarray]):
        batch_size = 8
        detections = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            # Predict detections for the current batch
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections.extend(detections_batch)
        
        return detections
    
    def get_object_tracks(self, frames: List[np.ndarray], read_from_stub=False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Detect objects in the frames
        detections = self.detect_frames(frames)

        tracks = {
            "player":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            # Get class names and map them to their IDs, but please invert it because it just mentally easier
            cls_names = detection.names
            cls_names_inverted = {v: k for k, v in cls_names.items()}
            
            # Convert YOLO detections to supervision Detections so we can actually do tracking for once.
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["player"].append({})
            tracks["ball"].append({})

            for frame_d in detections_with_tracks:
                bbox = frame_d[0].tolist()
                cls_id = frame_d[1]
                track_id = frame_d[2]
                if cls_id == cls_names_inverted['player']:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inverted['ball']:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}
                else:
                    print("Unknown class ID")
            
            if stub_path is not None:
                # Save the tracks to a file
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
            return tracks
    def draw_annotations(frame, detections):
        pass