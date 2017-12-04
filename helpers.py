from moviepy.editor import *
from obj_detection import *

def extract_frames(fname='./project_video.mp4', video_length_secs=50, interval_secs=1, 
        dst_dir='./test_frames/'):

    """ Extract some frames from a video clip and save then to disk """
    clip = VideoFileClip(fname)
    times = range(0, video_length_secs, interval_secs)
    for t in times:
        print(t)
        clip.save_frame(dst_dir + 'frame_{:03}.png'.format(t), t)

def save_sequence(fname='./project_video.mp4'):
    clip = VideoFileClip(fname)
    cnt = 0
    for frame in clip.iter_frames():
        imsave('./test_images/one_sequence/{}.png'.format(cnt), (frame - frame.min())[:,:,::-1])
        cnt += 1
        print(cnt)
        if cnt > 10:
            break

if __name__ == '__main__':
    #extract_frames()
    save_sequence()

