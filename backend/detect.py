import os
import argparse
import cv2
import beepy
import uuid

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


OUTPUT_DIR = 'outputs'


def load_model(model_path):
    return YOLO(model_path)

def draw_boxes(model, results, frame):
    for r in results:
        annotator = Annotator(frame)
        boxes = r.boxes
        warning_status = False
        color = (0, 255, 0) 

        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            label = model.names[int(c)]

            if label == 'head':
                color = (0, 0, 255)
                warning_status = True
                xmin, ymin, xmax, ymax = b
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                saved_frame = frame[ymin:ymax, xmin:xmax].copy()
                saved_frame = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)
                output_path = os.path.join(OUTPUT_DIR, str(uuid.uuid4()), "-ss.jpg")

                if not os.path.exists(output_path):
                    cv2.imwrite(output_path, saved_frame)
            else:
                if warning_status:
                    warning_status = False
                    color = (0, 255, 0) 

            if warning_status:
                beepy.beep(sound=1)

            annotator.box_label(b, model.names[int(c)], color)

def run_inference(model, cap, show_output):
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        if show_output:
            draw_boxes(model, results, frame)

            cv2.imshow('object_detection', cv2.resize(frame, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside video')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to video')
    parser.add_argument('-o', '--show_output', type=int, choices=[0, 1], required=False, default=1,  help='Show output with GUI')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    cap = cv2.VideoCapture(args.video_path)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    run_inference(detection_model, cap, bool(args.show_output))
