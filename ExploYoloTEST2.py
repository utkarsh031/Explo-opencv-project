# import cv2
# import argparse
# import time
# import serial
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# ZONE_POLYGON = np.array([
#     [0, 0],
#     [0.5, 0],
#     [0.5, 1],
#     [0, 1]
# ])


# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="YOLOv8 live")
#     parser.add_argument(
#         "--webcam-resolution", 
#         default=[1280, 720], 
#         nargs=2, 
#         type=int
#     )
#     args = parser.parse_args()
#     return args


# def main():
#     args = parse_arguments()
#     frame_width, frame_height = args.webcam_resolution

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#     model = YOLO("yolov8l.pt")

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

#     zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
#     zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
#     zone_annotator = sv.PolygonZoneAnnotator(
#         zone=zone, 
#         color=sv.Color.red(),
#         thickness=2,
#         text_thickness=4,
#         text_scale=2
#     )

#     arduino = serial.Serial('COM5', 9600)  # Change 'COM3' to your Arduino's port

#     start_time = None
#     while True:
#         if arduino.in_waiting > 0:
#             serial_input = arduino.readline().decode().strip()
#             if serial_input == '1':
#                 start_time = time.time()
        
#         if start_time is not None and time.time() - start_time <= 10:
#             ret, frame = cap.read()

#             result = model(frame, agnostic_nms=True)[0]
#             detections = sv.Detections.from_yolov8(result)
#             labels = [
#                 f"{model.model.names[class_id]} {confidence:0.2f}"
#                 for _, confidence, class_id, _
#                 in detections
#             ]
#             frame = box_annotator.annotate(
#                 scene=frame, 
#                 detections=detections, 
#                 labels=labels
#             )

#             zone.trigger(detections=detections)
#             frame = zone_annotator.annotate(scene=frame)      
            
#             cv2.imshow("yolov8", frame)

#         if (cv2.waitKey(30) == 27):
#             break


# if __name__ == "__main__":
#     main() 


# import cv2
# import argparse
# import time
# import serial
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np

# ZONE_POLYGON = np.array([
#     [0, 0],
#     [0.5, 0],
#     [0.5, 1],
#     [0, 1]
# ])

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="YOLOv8 live")
#     parser.add_argument(
#         "--webcam-resolution", 
#         default=[1280, 720], 
#         nargs=2, 
#         type=int
#     )
#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_arguments()
#     frame_width, frame_height = args.webcam_resolution

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#     model = YOLO("yolov8l.pt")

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

#     zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
#     zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
#     zone_annotator = sv.PolygonZoneAnnotator(
#         zone=zone, 
#         color=sv.Color.red(),
#         thickness=2,
#         text_thickness=4,
#         text_scale=2
#     )

#     arduino = serial.Serial('COM5', 9600)  

#     start_time = None
#     while True:
#         if arduino.in_waiting > 0:
#             serial_input = arduino.readline().decode().strip()
#             if serial_input == '1':
#                 start_time = time.time()
        
#         if start_time is not None and time.time() - start_time <= 10:
#             ret, frame = cap.read()

#             result = model(frame, agnostic_nms=True)[0]
#             detections = sv.Detections.from_yolov8(result)
#             labels = [
#                 f"{model.model.names[class_id]} {confidence:0.2f}"
#                 for _, confidence, class_id, _
#                 in detections
#             ]
#             frame = box_annotator.annotate(
#                 scene=frame, 
#                 detections=detections, 
#                 labels=labels
#             )

#             zone.trigger(detections=detections)
#             frame = zone_annotator.annotate(scene=frame)      
            
#             cv2.imshow("yolov8", frame)

#         else:
#             cv2.destroyAllWindows()  
#         if cv2.waitKey(30) == 27:
#             break

#     # Release resources
#     cap.release()
#     arduino.close()

# if __name__ == "__main__":
#     main()

# import cv2
# import argparse
# import time
# import serial
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np

# ZONE_POLYGON = np.array([
#     [0, 0],
#     [0.5, 0],
#     [0.5, 1],
#     [0, 1]
# ])

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="YOLOv8 live")
#     parser.add_argument(
#         "--webcam-resolution", 
#         default=[1280, 720], 
#         nargs=2, 
#         type=int
#     )
#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_arguments()
#     frame_width, frame_height = args.webcam_resolution

#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#     model = YOLO("yolov8l.pt")

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

#     zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
#     zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
#     zone_annotator = sv.PolygonZoneAnnotator(
#         zone=zone, 
#         color=sv.Color.red(),
#         thickness=2,
#         text_thickness=4,
#         text_scale=2
#     )

#     arduino = serial.Serial('COM5', 9600)  

#     previous_serial_input = None
#     detection_active = False
#     detection_start_time = None

#     while True:
#         if arduino.in_waiting > 0:
#             current_serial_input = arduino.readline().decode().strip()
#             if previous_serial_input != current_serial_input:
#                 if current_serial_input == '1' or current_serial_input == '0':
#                     if not detection_active:  # If detection is not already active
#                         detection_active = True
#                         detection_start_time = time.time()
#             previous_serial_input = current_serial_input

#         if detection_active and (detection_start_time is None or time.time() - detection_start_time <= 10):
#             ret, frame = cap.read()

#             result = model(frame, agnostic_nms=True)[0]
#             detections = sv.Detections.from_yolov8(result)
#             labels = [
#                 f"{model.model.names[class_id]} {confidence:0.2f}"
#                 for _, confidence, class_id, _
#                 in detections
#             ]
#             frame = box_annotator.annotate(
#                 scene=frame, 
#                 detections=detections, 
#                 labels=labels
#             )

#             zone.trigger(detections=detections)
#             frame = zone_annotator.annotate(scene=frame)      
            
#             cv2.imshow("yolov8", frame)

#         else:
#             cv2.destroyAllWindows()  

#         if cv2.waitKey(30) == 27:
#             break

#     # Release resources
#     cap.release()
#     arduino.close()

# if __name__ == "__main__":
#     main()


import cv2
import argparse
import time
import serial
from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    arduino = serial.Serial('COM5', 9600)  

    previous_serial_input = None
    detection_active = False
    detection_start_time = None

    while True:
        if arduino.in_waiting > 0:
            current_serial_input = arduino.readline().decode().strip()
            if previous_serial_input != current_serial_input:
                if current_serial_input == '1' or current_serial_input == '0':
                    if not detection_active:  # If detection is not already active
                        detection_active = True
                        detection_start_time = time.time()
            previous_serial_input = current_serial_input

        if detection_active:
            start_time = time.time()
            while time.time() - start_time <= 10:  # Run detection for 10 seconds
                ret, frame = cap.read()

                result = model(frame, agnostic_nms=True)[0]
                detections = sv.Detections.from_yolov8(result)
                labels = [
                    f"{model.model.names[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, _
                    in detections
                ]
                frame = box_annotator.annotate(
                    scene=frame, 
                    detections=detections, 
                    labels=labels
                )

                zone.trigger(detections=detections)
                frame = zone_annotator.annotate(scene=frame)      
                
                cv2.imshow("yolov8", frame)

                if cv2.waitKey(30) == 27:
                    break

            detection_active = False  # Reset detection_active after 10 seconds

        else:
            cv2.destroyAllWindows()  

            if cv2.waitKey(30) == 27:
                break

    # Release resources
    cap.release()
    arduino.close()

if __name__ == "__main__":
    main()
