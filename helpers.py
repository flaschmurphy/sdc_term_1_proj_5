import matplotlib.pyplot as plt
from moviepy.editor import *
from obj_detection import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--classifier', dest='clf',
            help='The location of the pickled classifier')

    args = parser.parse_args()
    return args


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

def explore_features():
    """Plot the decision function for a few images"""
    assert args.clf is not None, 'Must supply a classifier in args'

    car_fnames = glob.glob('./test_images/car_*') 
    notcar_fnames = glob.glob('./test_images/notcar_*')

    clf = pickle.load(open(args.clf, 'rb'))
    coefs = clf.coef_[0]

    car_features = []
    notcar_features = []
    cnt = 0

    # Generate the input data for the pipeline
    car_img_data = []
    for fname in car_fnames:
        img = imread(fname, for_prediction=True)
        img_data_this = [fname, img, cnt]
        car_img_data.append(img_data_this)
        cnt += 1

    car_preds = []
    for fname in car_fnames:
        scaler_fname = ''.join(args.clf.split('.')[:-1]) + '_scaler.pkl'
        img = imread(fname, for_prediction=True)
        car_preds.append(predict(img, clf, scaler_fname))
    
    notcar_img_data = []
    for fname in notcar_fnames:
        img = imread(fname, for_prediction=True)
        img_data_this = [fname, img, cnt]
        notcar_img_data.append(img_data_this)
        cnt += 1

    notcar_preds = []
    for fname in notcar_fnames:
        scaler_fname = ''.join(args.clf.split('.')[:-1]) + '_scaler.pkl'
        img = imread(fname, for_prediction=True)
        notcar_preds.append(predict(img, clf, scaler_fname))

    car_features = pipeline(car_img_data)
    notcar_features = pipeline(notcar_img_data)

    carpts = np.dot(car_features, coefs)
    print('Car data:')
    print(carpts)
    print(car_preds)

    notcarpts = np.dot(notcar_features, coefs)
    print('Not car data:')
    print(notcarpts)
    print(notcar_preds)


if __name__ == '__main__':
    args = parse_args()
    #extract_frames()
    #save_sequence()
    #explore_features()
    pass

