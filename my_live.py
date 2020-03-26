from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.utils.misc import Timer, str2bool
import torch
import cv2
import sys
import argparse
import time

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--model_path", default="./models/vgg16-ssd-mp-0_7726.pth",type=str)

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--label_path",default="./models/voc-model-labels.txt", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

class_names = [name.strip() for name in open(args.label_path).readlines()]
num_classes = len(class_names)

if args.net == 'vgg16-ssd':
    net = create_vgg_ssd(num_classes, is_test=True)
    net.load(args.model_path)
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=torch.device("cpu"))
elif args.net == "mb2-ssd-lite":
    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
    net.load(args.model_path)

    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200,
                                                      device=torch.device('cuda'))

timer = Timer()
cap = cv2.VideoCapture("./test_video/test_video_traffic.mp4")  # capture from file
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#video = cv2.VideoWriter("./mobilenet_v2.avi", fourcc, 30.0, (534, 300))

#start = time.time()

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    #video.write(orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #if time.time() - start > 15:
    #    break

cap.release()
#video.release()
cv2.destroyAllWindows()
