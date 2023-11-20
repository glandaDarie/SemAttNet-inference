from typing import List, Tuple
import cv2
import os
from re import search
import numpy as np

def extract_numeric_part(filename : str) -> int:
    """
    Extracts numeric part from a filename using regex.

    Args:
    - filename (str): The filename from which to extract numeric part.

    Returns:
    - int: Extracted numeric part or -1 if not found.
    """
    match = search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1

def get_image_files_and_directories(current_dir : str , path_targets) -> Tuple[List[str], List[str]]:
    """
    Collects image files and directories based on path_targets.

    Args:
    - directory (str): Base directory path.
    - path_targets (list): List of subdirectories to explore.

    Returns:
    - list: List of image files for each path_target.
    - list: List of directories for each path_target.
    """
    directories = []
    targets = []
    for path_target in path_targets:
        dir_path = os.path.join(current_dir, "inference_helpers", "data_depth_selection",
                                "data_depth_selection", "val_selection_cropped", path_target)
        image_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        image_files.sort(key=lambda x: extract_numeric_part(x))
        targets.append(image_files)
        directories.append(dir_path)
    return targets, directories


def concatenate_images(targets : List[List[str]], directories : List[str]) -> List[np.ndarray]:
    """
    Concatenates images based on given targets and directories.

    Args:
    - targets (list(list)): List of image files for each path_target.
    - directories (list): List of directories for each path_target.
    
    Returns:
    - list: List of concatenated images.
    """
    concatenated_images = []
    max_width = 0
    max_height = 0
    for images in zip(*targets):
        concatenated_image = None
        images_to_concat = []
        for image_name, directory in zip(images, directories):
            image_path = os.path.join(directory, image_name)
            current_img = cv2.imread(image_path)
            if current_img is not None:
                images_to_concat.append(current_img)
                max_width = max(max_width, current_img.shape[1] // 2)
                max_height = max(max_height, current_img.shape[0] + 100)
        for i in range(len(images_to_concat)):
            images_to_concat[i] = cv2.resize(images_to_concat[i], (max_width, max_height))
        concatenated_image = np.concatenate(images_to_concat, axis=0) if len(images_to_concat) > 0 else None
        concatenated_images.append(concatenated_image)
    return concatenated_images

def display_concatenated_images(concatenated_images : List[np.ndarray], save_video : bool = False) -> None:
    """
    Displays concatenated images.

    Args:
    - concatenated_images (list): List of concatenated images.
    - save_video (bool): if it should save a video of the depth completion
    """
    for concatenated_image in concatenated_images:
        if concatenated_image is not None:
            cv2.imshow("KITTI", concatenated_image)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    if save_video:
        generate_video(concatenated_images=concatenated_images)

def generate_video(concatenated_images, video_location=r"C:\Users\darie\OneDrive\Desktop", video_name="depth_completion.mp4") -> None:
    """
    Saves concatenated images as a video.

    Args:
    - concatenated_images (list): List of concatenated images.
    - video_location (str): Folder path where the video will be saved.
    - video_name (str): Name of the video file.
    """
    if concatenated_images:
        filename = os.path.join(video_location, video_name)
        height, width, _ = concatenated_images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=15, frameSize=(width, height))
        for concatenated_image in concatenated_images:
            if concatenated_image is not None:
                writer.write(concatenated_image)
        writer.release()
        print(f"Video saved in location: {filename}")

if __name__ == "__main__":
    current_directory = os.getcwd()
    path_targets = [
        "image",
        "depth_completion",
    ]
    targets, directories = get_image_files_and_directories(current_dir=current_directory, path_targets=path_targets)
    concatenated_images = concatenate_images(targets=targets, directories=directories)
    display_concatenated_images(concatenated_images=concatenated_images, save_video=True)