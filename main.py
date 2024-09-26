from utils import read_video, save_video
from trackers.tracker import Tracker
from camera_mov_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speedanddistance import SpeedAndDistance

def main():
    #Read Video
    video_frames = read_video('NFL_Clips/clip2.mov')

    #Initialize Tracker
    tracker = Tracker('models/best-3.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
    #get object positions
    tracker.add_pos_to_tracks(tracks)
    #Camera Movement Estimation
    camera_movement_estimation = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frames = camera_movement_estimation.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    view_transformer = ViewTransformer()
    view_transformer.transform(tracks)

    #Adjust object positions to camera movement
    tracker._adjust_pos_to_tracks(tracks, camera_movement_per_frames)
    
    #interpolate ball positions
    tracks['football'] = tracker.interpolate_ball_pos(tracks['football'])

    #Add speed and distance to tracks
    speed_and_distance = SpeedAndDistance()
    speed_and_distance.add_speed_and_distance_to_track(tracks)

    output_vid_frames = tracker.draw_annotations(video_frames, tracks)

    output_vid_frames = camera_movement_estimation.draw_camera_movement(output_vid_frames, camera_movement_per_frames)

    speed_and_distance.draw_speed_and_distance(output_vid_frames, tracks)

    #Save Video
    save_video(output_vid_frames, 'output_clips/output_vid.avi')

if __name__ == "__main__":
    main()