from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import collections
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=2,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./cfg/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    if (orig_left < 0):
        orig_left = 0
    if (orig_right > image_w - 1):
        orig_right = image_w - 1
    if (orig_top < 0):
        orig_top = 0
    if (orig_bottom > image_h - 1):
        orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    # yong, 2021.05.11
    # 기록된 정보 저장용 장소, 시간 측정
    my_prev_time = None
    temp = []
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)

        # yong, 2021.05.11
        # 3초마다 기록된 정보를 test.txt에 써줌
        with open('test.txt', 'a') as f:
            if my_prev_time == None or time.time() - my_prev_time > 3:
                print('yong: recording...')
                for t in temp:
                    f.write(
                        ''.join([str(d) + ':' for d in t]) + '\n')
                temp.clear()
                my_prev_time = time.time()
            else:
                temp.append(detections)

        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        # print("FPS: {}".format(fps))
        # darknet.print_detections(detections, True)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    # video = set_saved_video(cap, args.out_filename,
    #                         (darknet_width, darknet_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append(
                    (str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(
                detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # if args.out_filename is not None:
            #     video.write(image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    # video.release()
    cv2.destroyAllWindows()


def myTestFunction():
    def checkAllIn():
        all = []
        for t in temp:
            for _t in t.split(':'):
                name = _t.split(',')[0].replace('\'', '').replace('(', '')
                if name != '\n':
                    temp_dict[name].append(_t)
        for td in temp_dict:
            print(len(temp), len(temp_dict[td]), td)
            if len(temp_dict[td]) == len(temp):
                all.append(td)
        return all

    def checkPair(person):
        person = tuple(person)
        pair = [person, []]
        flag = True
        for a in all:
            if a != 'person':
                for i in range(len(temp)):
                    temp_person = [p.replace('\'', '').replace('(', '').replace(')', '').strip()
                                   for p in person[i].split(',')]
                    temp_person = [float(tp) if tp.replace('.', '').isdigit() else tp
                                   for tp in temp_person]
                    temp_other = [o.replace('\'', '').replace('(', '').replace(')', '').strip()
                                  for o in temp_dict[a][i].split(',')]
                    temp_other = [float(to) if to.replace('.', '').isdigit() else to
                                  for to in temp_other]
                    # print(temp_person)
                    # print(type(temp_person))
                    # print(temp_other)
                    # print(type(temp_other))
                    # print(np.sqrt(np.square(
                    #     temp_person[2] - temp_other[2]) + np.square(temp_person[3] - temp_other[3])))
                    # print((temp_person[4] + temp_person[5]) // 2 // 10)
                    if np.sqrt(np.square(temp_person[2] - temp_other[2]) + np.square(temp_person[3] - temp_other[3])) \
                            > (temp_person[4] + temp_person[5]) // 2:
                        flag = False
                if flag:
                    pair[1].append(a)
        return pair if len(pair[1]) > 0 else []

    my_prev_time = None
    pairs = []
    while cap.isOpened():
        # yong, 2021.05.11
        # 3초마다 기록된 정보를 test.txt에서 가져옴
        if my_prev_time == None or time.time() - my_prev_time > 3:
            with open('test.txt', 'r') as f:
                print('yong: reading...')
                temp = f.readlines()
                temp_dict = collections.defaultdict(list)
                all = checkAllIn()
                # print('all ->', all)
                people = []
                for a in all:
                    if a == 'person':
                        people.append(temp_dict[a])
                # print('people ->', people)
                pairs = []
                for p in people:
                    pairs.append(checkPair(p))
                print('pairs ->', pairs)
                # print('items ->')
                # for t in temp:
                #     print(t.split(',')[0].replace('(', ''), end=' ')
                my_prev_time = time.time()
            with open('test.txt', 'w') as f:
                f.write('')


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    with open('test.txt', 'w') as f:
        f.write('')
    Thread(target=video_capture, args=(
        frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue,
           detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue,
           detections_queue, fps_queue)).start()
    Thread(target=myTestFunction).start()
