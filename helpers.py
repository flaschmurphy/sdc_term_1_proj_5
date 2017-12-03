from moviepy.editor import *

def extract_frames(fname='./project_video.mp4', video_length_secs=50, interval_secs=1, 
        dst_dir='./test_frames/'):

    """ Extract some frames from a video clip and save then to disk """
    clip = VideoFileClip(fname)
    times = range(0, video_length_secs, interval_secs)
    for t in times:
        print(t)
        clip.save_frame(dst_dir + 'frame_{:03}.png'.format(t), t)

if __name__ == '__main__':
    extract_frames()
