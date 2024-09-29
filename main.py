from utils import read_video, save_video

from trackers import PlayerTracker

def main():

    #Read the video
    input_video_path = 'input_videos/input_video.mp4'

    video_frames = read_video(input_video_path)


    #detecting the players
    player_tracker = PlayerTracker(model_path='yolov8x')

    player_detections = player_tracker.detect_frame(video_frames)

    save_video(video_frames, "output_videos/output_video.avi")
    
    
    print("Hello, World!")

if __name__ == '__main__':
        main()