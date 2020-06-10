import sys, argparse
from feed import IO
from structure_detect import Face
from algorithm import EARtracker
from playsound import playsound
import time


class API:
    def __init__(self, args):
        self.args = args
        if args.src == 'webcam':
            self.io = IO(feed_option=self.args.src, webcam_id=self.args.cam_id, frame_size=self.args.frame_size, show_frames=self.args.vis, plot_act=self.args.graph)
        elif args.src == 'rec_video':
            self.io = IO(feed_option=self.args.src, file_path=self.args.file_path, frame_size=self.args.frame_size, show_frames=self.args.vis, plot_act=self.args.graph)
        elif args.src == 'image':
            self.io = IO(feed_option=self.args.src, file_path=self.args.file_path, frame_size=self.args.frame_size, show_frames=self.args.vis, plot_act=self.args.graph)
        else:
            print('What other input source?')
            exit(0)

        self.face = Face(model=args.model)

        self.algorithm = EARtracker()

    def run(self):
        frame = self.io.next_frame(face_box=None, eye_boxes=None, structure_pts=None)
        timeflag=1  #flag for if face is detected or not
        redflag=0 #flag for making red if face is not detected for > 3 sec
        while frame is not None:
            if timeflag==1 or flag==1:
                t1=time.time()
                redflag=0
            elif timeflag==0 or flag==0:
                t2=time.time()
                if t2-t1>3:
                    redflag=1
                    #playsound('../sound/warning_sound.mp3')  programm crashes
                else:
                    redflag=0

                    
            face_box, eye_boxes, structure_pts, timeflag = self.face.extract(frame)
            score, flag = self.algorithm.process(face_box, eye_boxes, structure_pts)
            frame = self.io.next_frame(face_box=face_box, eye_boxes=eye_boxes, structure_pts=structure_pts, flag=flag, score=score, redflag=redflag)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", required=True, choices=['webcam', 'rec_video', 'image'],
    	help="select the source from ['webcam', 'rec_video', 'image']")
    parser.add_argument("-p", "--file_path", type=str, default=None,
    	help="path for video/image file")
    parser.add_argument("-i", "--cam_id", type=int, default=0,
    	help="index of webcam on system")
    parser.add_argument("-w", "--frame_size", type=int, default=450,
    	help="frame size for the video")
    parser.add_argument("-m", "--model", type=str, default='../models/68_landmark_model.dat',
    	help="path to detector model")
    parser.add_argument("-v", "--vis", type=bool, default=True,
    	help="show frames and tracked structures")
    parser.add_argument("-g", "--graph", type=bool, default=True,
    	help="plot the activeness measeure for the driver wrt time")
    args = parser.parse_args()

    api = API(args)
    api.run()
