"""
Created on Sun Nov 21 12:34:50 2021
@author: Manuel J. Corbacho
"""
import torch
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio #etract images form videos

#Define a function to detect a dog in a image
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.8:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame
#Create pretrained neural network
ssd_network = build_ssd('test')
# get the file ssd300_mAP_77.43_v2 on https://github.com/anushuk/Object-Detection-SSD
ssd_network.load_state_dict(
    torch.load(
        'ssd300_mAP_77.43_v2.pth', 
        map_location = lambda storage, 
        loc: storage
    )
)

#create the transformation with the proper scale for the pretrained ssd
my_transform = BaseTransform(ssd_network.size, (104/256.0, 117/256.0, 123/256.0) )

#detect on video
reader = imageio.get_reader('name_of_a_video_file.mp4')
input_fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = input_fps)
for i, frame in enumerate(reader):
    frame = detect(frame, ssd_network.eval(), my_transform)
    writer.append_data(frame)
    print(i)
writer.close()