from itertools import count
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms
import numpy as np

def detect(model, input_file, output_file, fps=30, score_filter=0.7):
    video = cv2.VideoCapture(input_file)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scaled_size = 1600
    scale_down_factor = min(frame_height, frame_width) / scaled_size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))
    transform_frame = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(scaled_size),
        transforms.ToTensor(),
        normalize_transform(),
    ])
    count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        transformed_frame = frame  # TODO: Issue #16
        predictions = model.predict(transformed_frame)
        print(predictions)
        # Add the top prediction of each class to the frame
        for label, box, score in zip(*predictions):
            if score < score_filter:
                continue 
            print("Score: ",score)           
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(frame, c1, c2, (0, 255, 0), 3)
            img = frame[c1[1]:c2[1],c1[0]:c2[0]]
            cv2.imwrite('output/'+str(count)+'.jpg', img)
            count += 1
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When finished, release the video capture and writer objects
    video.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()


# from detecto.core import Model
# from detecto import utils, visualize

# model = Model.load('model.pth', ['license'])
# image = utils.read_image('frame2.jpg')  # Helper function to read in images
# labels, boxes, scores = model.predict(image)  # Get all predictions on an image
# predictions = model.predict_top(image)  # Same as above, but returns only the top predictions
# print(labels, boxes, scores)
# print(predictions)
# detect(model, labels, boxes, scores, '2.mp4', 'output.mp4')
# print('completed')