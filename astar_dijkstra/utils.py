import cv2
import pygetwindow as gw
import numpy as np
import time
import os
from PIL import ImageGrab

def record_window_to_video(window_title, video_name, duration, frame_rate=30):
    """
    Records the specified window to a video file.
    
    :param window_title: The title of the window to capture.
    :param video_name: Name of the output video file.
    :param duration: Duration of the video in seconds.
    :param frame_rate: Frame rate of the output video.
    """
    # Ensure the directory for saving the video exists
    output_dir = './video'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Construct the full path to the video file
    video_path = os.path.join(output_dir, video_name)

    # Find the window by title
    window = None
    while window is None:
        windows = gw.getWindowsWithTitle(window_title)
        if windows:
            window = windows[0]  # Select the first matching window
            if window.isMinimized:  # Minimized windows can't be captured
                print(f"Window {window_title} is minimized. Waiting...")
                time.sleep(1)
                continue
            break
        else:
            print(f"Window {window_title} not found,retrying")
            time.sleep(0.5)

    # Get window geometry (left, top, right, bottom)
    left, top, right, bottom = window.left, window.top, window.right, window.bottom

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (right - left, bottom - top))

    start_time = time.time()
    while time.time() - start_time < duration:
        # Capture the screen content of the window
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame to video file
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to {video_path}")
