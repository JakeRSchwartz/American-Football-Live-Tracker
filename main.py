from utils import read_video, save_video
from trackers.tracker import Tracker

def main():
    #Read Video
    video_frames = read_video('NFL_Clips/clip2.mov')

    #Initialize Tracker
    tracker = Tracker('models/best-3.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    output_vid_frames = tracker.draw_annotations(video_frames, tracks)

    #Save Video
    save_video(output_vid_frames, 'output_clips/output_vid.avi')

if __name__ == "__main__":
    main()