import numpy as np
import cv2
class ViewTransformer():
    def __init__(self):
        field_width = 48.8
        field_height = 27.4


        self.pixel_verticies = np.array([
            [110, 1025],
            [250, 274],
            [900,250],
            [1650, 900],
        ])

        self.target_verticies = np.array([
            [0, field_width],
            [0, 0],
            [field_height, 0],
            [field_height,  field_width],
        ])

        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies)


    def transform_point(self, point):
        p = (int(point[0], int(point[1])))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies, p, False) >= 0
        if not is_inside:
            return None
        reshape_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshape_point, self.perspective_transformer)

        return transformed_point.reshape(-1,2)
    
    def transform(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted', [])
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
