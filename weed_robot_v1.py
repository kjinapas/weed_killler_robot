import time
import serial
import struct
from collections import defaultdict
import cv2
import os
import numpy as np
from ultralytics import YOLO
import threading
import sys  # Import sys to use sys.exit()

mask_color = (255, 0, 0)  # Blue color for mask
text_color = (0, 0, 0)  # Black color for text
bg_color = (0, 255, 255)  # Yellow color for text background
grid_color = (255, 255, 255)  # White color for grid lines
subline_color = (128, 128, 128)  # Gray color for sublines
bbox_color = (0, 255, 0)  # Green color for bounding box

grid_spacing = 100  
subline_spacing = 10
output_folder_path = 'output'
os.makedirs(output_folder_path, exist_ok=True)
# Configure the serial ports
try:
    serial_To_stm32_1 = serial.Serial('COM11', baudrate=115200, timeout=0.1)
    serial_To_stm32_2 = serial.Serial('COM10', baudrate=115200, timeout=0.1)#--gripper----#
    print("Serial ports opened successfully")
except Exception as e:
    print(f"Error opening serial ports: {e}")
    sys.exit(1)

# Control the gripper to position that will be ready for taking video
# String to send
delta_init_position = "0,60,-120\n"
serial_To_stm32_2.write(delta_init_position.encode())  # Encode the string to bytes

# Load the saved calibration parameters
calibration_data = np.load("camera_calibration.npz")
camera_matrix = calibration_data['camera_matrix']
distortion_coefficients = calibration_data['dist_coeffs']
# brightness_scale = 0.75

# Delta robot parameters
height_cam = 320
pix_center_x = 297/2
pix_center_y = 100 # 228/2

delta_x_scale = (340/297/320)*height_cam
delta_y_scale = (244/228/320)*height_cam
delta_x_offset = -22 -40
delta_y_offset = -93 + 70
delta_pick_height = -230
head_step = 42

# Limits delta
delta_x_min = -130
delta_x_max = 130
delta_y_min = -130
delta_y_max = 130

# If condition of test
# IF_MOVE_HEAD = False
IF_MOVE_HEAD = True

# Load the YOLOv8 model
model = YOLO('best__weedmodel_student.pt')
# model = YOLO('weed-detection-3000-epoch.pt')

# Open the video file
video_path = 0 # Change to your video path if not using a camera// camera4
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    exit()



# Store the track history
track_history = defaultdict(lambda: [])

# Constants
z_data_stm32_2 = -320
address_stm32_1, address_stm32_2 = 72, 72

# Initialize limit trigger states
limit_triggered = {'left': False, 'right': False}
start_processing = False
# allez = False  # Initialize allez variable
# Initialize movement direction
movement_direction = head_step  # Start with 50 for right direction
base_direction = 1500
# Initialize camera position on x-axis
camera_position_x = 0

# Limits for x-coordinates
# x_min_limit = 160
# x_max_limit = 1200

# count number limit_r reached
count_r = 0

def check_axis_limit(x,y):
    global delta_x_min
    global delta_x_max
    global delta_y_min
    global delta_y_max
    if x < delta_x_min:
        return False
    elif x > delta_x_max:
        return False
    elif y < delta_y_min:
        return False
    elif y > delta_y_max:
        return False
    else:
        return True
            
# def adjust_brightness(frame, brightness_factor):
#     # Convert the frame to a float32 array for precise calculations
#     frame = frame.astype('float32')
    
#     # Adjust brightness
#     frame = frame * brightness_factor
    
#     # Clip the values to stay within valid range [0, 255] and convert back to uint8
#     frame = np.clip(frame, 0, 255).astype('uint8')
    
#     return frame

def send_data_to_stm32(serial_port, address, data):
    try:
        to_stm32 = [[address], [data]]
        flat_data_stm32 = [item for sublist in to_stm32 for item in sublist]
        packed_data_stm32 = struct.pack(f'{len(flat_data_stm32)}i', *flat_data_stm32)
        serial_port.write(packed_data_stm32)
        # print(f"Data sent to STM32: address = {address}, data = {data}")

    except Exception as e:
        print(f"Error sending data: {e}")

def send_speed_and_stop(serial_port, address, speed, duration):
    send_data_to_stm32(serial_port, address, speed)
    time.sleep(duration)
    send_data_to_stm32(serial_port, address, 0)  # Stop the movement

def check_limit(serial_port):
    while True:
        try:
            if serial_port.in_waiting > 0:
                char = serial_port.read(1).decode('utf-8')
                print(f"Character received: {char}")
                handle_limit(char)
        except Exception as e:
            print(f"Error reading data: {e}")

def handle_limit(char):
    global start_processing
    global movement_direction
    global camera_position_x
    global base_direction
    global limit_triggered 
    global count_r 

    if char == 'L':
        limit_triggered['left'] = True
        print("Left limit triggered")
        camera_position_x = 1200  # Approximate position when left limit is reached
        movement_direction = -head_step  # Change direction to left
        print("Movement direction:", movement_direction)
        
    elif char == 'R':
        limit_triggered['right'] = True
        count_r += 1 
        print("Right limit triggered")
        print("count_r = " , count_r)
        camera_position_x = 0  # Reset position to 0 when right limit is reached
        movement_direction = head_step  # Change direction to right
        time.sleep(2)
        start_processing = True
        print("Movement direction:", movement_direction)
    print("Camera position x:", camera_position_x)

def draw_graduated_axis(frame, position, reference_y):
    height, width = frame.shape[:2]
    step = 50
    offset = 0  # Offset value to adjust downwards 

    for x in range(0, width, step):
        label = str(position + x - width // 2)
        cv2.putText(frame, label, (x, reference_y + offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(frame, (x, reference_y + 5), (x, reference_y + offset - 10), (255, 255, 255), 2)

    return frame


def process_frameA(frame):
    # results = model.track(frame, verbose=False, conf=0.50, persist=True, classes=[4], max_det=2) # 4 is weed, 5 is veg
    # results = model.track(frame, verbose=False, conf=0.70, persist=True, classes=[4, 5], max_det=2) # 4 is weed, 5 is veg    
    
    results = model.track(frame, verbose=False, conf=0.4, persist=True,max_det = 2)
    # cv2.line(frame, (0, 320//2), (640, 320//2), (0, 0, 255), 1)
    # cv2.line(frame, (640//2, 0), (640//2, 640), (0, 0, 255), 1)
    


    #increace by ton draw grid###
    height, width = frame.shape[:2]
    for x in range(0, width, grid_spacing):
        cv2.line(frame, (x, 0), (x, height), grid_color, 1)
        # Draw sublines within the major grid lines
        for sub_x in range(x + subline_spacing, x + grid_spacing, subline_spacing):
            cv2.line(frame, (sub_x, 0), (sub_x, height), subline_color, 1)
    for y in range(0, height, grid_spacing):
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
        # Draw sublines within the major grid lines
        for sub_y in range(y + subline_spacing, y + grid_spacing, subline_spacing):
            cv2.line(frame, (0, sub_y), (width, sub_y), subline_color, 1)


    print(results)

    print("Weed found: ", len(results[0].boxes.cls))

    if len(results[0].boxes.cls) > 0:
        # frame = draw_graduated_axis(frame, camera_position_x, 150)
        time.sleep(2)

        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.cls.int().cpu().tolist()

        frame = results[0].plot(masks=False)

        # Sort boxes by their x-coordinate
        sorted_indices = np.argsort(boxes[:, 0])
        boxes = boxes[sorted_indices]
        track_ids = [track_ids[i] for i in sorted_indices]

        iii=1

        for order, (box, track_id) in enumerate(zip(boxes, track_ids), start=1):

            print(iii)
            x, y, w, h = box

            track = track_history[track_id]
            track.append((float(x), float(y)))

            # adjusted_x = camera_position_x + (x - 320)  # Adjust x to be relative to the global position
            x_pix = x+w/2-pix_center_x # reletive to zeros at center
            y_pix = -(y+h/2-pix_center_y) # reletive to zeros at center
            x_delta = x_pix*delta_x_scale + delta_x_offset
            y_delta = y_pix*delta_y_scale + delta_y_offset

            print("X_pix: ", x_pix, "Y_pix: ", y_pix)
            print(f"x: {x}, y: {y}, w:{w}, h:{h} ")
            print(f'delta position: x {x_delta} ,y {y_delta}')
            
            
            if (check_axis_limit(x_delta, y_delta)):
                
                print("Yes! It is within the range")

                if len(track) > 1:
                    track.pop(0)

                points = np.hstack(track[0]).astype(np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, tuple(points[0][0]), 3, (0, 0, 255), 10)

                cv2.imshow("YOLOv8 Tracking", frame)

                # Annotate the frame with order and x position
                cv2.putText(frame, f'Order: {order}, X: {int(x)}', (int(x - w/2), int(y - h/2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                
                # position0 = "0,60,-250\n"
                z_delta = delta_pick_height + 50
                position1 = str(int(x_delta)) + "," + str(int(y_delta)) + "," + str(int(z_delta))  + "\n"
                print("position1", position1)
                serial_To_stm32_2.write(position1.encode())
                time.sleep(3)

                z_delta = delta_pick_height
                position2 = str(int(x_delta)) + "," + str(int(y_delta)) + "," + str(int(z_delta))  + "\n"
                print("position2", position2)
                serial_To_stm32_2.write(position2.encode())
                time.sleep(3)

                serial_To_stm32_2.write(position1.encode())
                time.sleep(3)

                # z_delta = delta_pick_height + 50
                # position3 = str(int(x_delta-75)) + "," + str(int(y_delta)) + "," + str(int(z_delta))  + "\n"
                # serial_To_stm32_2.write(position3.encode())
                # time.sleep(2)

                serial_To_stm32_2.write(delta_init_position.encode())
                time.sleep(2)

           
                # CALIBRATE HERE
            else:
               
                print("The weed found is out of manipulator range")
               
                
            # else:
            #     cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 2)
            #     cv2.putText(frame, f'Out of bounds: {int(adjusted_x)}', (int(x - w/2), int(y - h/2) - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            iii= iii+1

    return frame

def main():
    global camera_position_x  # Declare camera_position_x as global
    global limit_triggered  # Declare limit_triggered as global
    global iii
    limit_thread = threading.Thread(target=check_limit, args=(serial_To_stm32_1,))
    limit_thread.daemon = True
    limit_thread.start()
 
    # while not start_processing:
    #     time.sleep(0.1)

    while True:
        print("Opening Web Camera...")
        time.sleep(1)
        
        detect_cnt = 0
        undetect_cnt = 0
        # for frame_cnt in range(1):
        if cap.isOpened():
            success, frame = cap.read()
            h,w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
            if not success:
                print("Failed to capture frame")
                break

            undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, new_camera_matrix)
            # Crop the frame (if needed) based on the roi
            x, y, w, h = roi
            # undistorted_frame = undistorted_frame[y:y+h, x:x+w]
            frame = undistorted_frame[y:y+h, x:x+w]

            # Adjust birghtness
            matrix = np.ones(frame.shape, dtype="uint8") * 60
            # Increase the intensity of the frame using cv2.add()
            frame = cv2.subtract(frame, matrix)


            # processed_frame = process_frameA(frame, track_history)   
            processed_frame = process_frameA(frame)   
            cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_AUTOSIZE)
            # cv2.resizeWindow("YOLOv8 Tracking", 640, 480)
            # cv2.resizeWindow("YOLOv8 Tracking", 480, 640)
            cv2.imshow("YOLOv8 Tracking", processed_frame)

            if len(track_history) == 0:
                undetect_cnt += 1

        # Now move the manipulator base
        if limit_triggered['left']:
            send_speed_and_stop(serial_To_stm32_1, 66, base_direction, 5)
            limit_triggered['left'] = False  # Reset after sending command
            # time.sleep(3)

        # elif limit_triggered['right'] and count_r == 2:
        elif limit_triggered['right']:
            send_speed_and_stop(serial_To_stm32_1, 66,base_direction, 5)
            limit_triggered['right'] = False
        
        if IF_MOVE_HEAD:
            send_data_to_stm32(serial_To_stm32_1, 72, movement_direction)
        
        time.sleep(3)

        camera_position_x += movement_direction
        
        # limit_triggered = {'left': False, 'right': False}

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User pressed 'q', exiting main loop")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
