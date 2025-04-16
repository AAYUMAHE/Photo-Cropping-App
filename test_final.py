import cv2
import mediapipe as mp
import os
import shutil
import numpy as np

#this script is also considering the , faces with tilted photos.

#updated by me , after neck padding increased .

#and after adding the missing files details . 

def detect_face(image, face_detection):
    """Run face detection and return bounding box if found."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            return x1, y1, x2, y2  # Face detected

    return None  # No face detected

def rotate_image(image, angle):
    """Rotate image by 90° or -90°."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def crop_passport_photos(input_folder, output_folder, no_face_folder):
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(no_face_folder, exist_ok=True)

    # Track skipped files
    skipped_files = []

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-image files, but move them for consistency
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            skipped_files.append((filename, "Unsupported format"))
            shutil.move(input_path, os.path.join(no_face_folder, filename))
            continue

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            skipped_files.append((filename, "Unreadable image"))
            shutil.move(input_path, os.path.join(no_face_folder, filename))
            print(f"Skipped unreadable: {filename}")
            continue

        # Try detecting the face in the original image
        face_bbox = detect_face(image, face_detection)

        # If no face is found, try rotated versions (90°, -90°, 180°)
        if face_bbox is None:
            for angle in [90, -90, 180]:
                rotated_image = rotate_image(image, angle)
                face_bbox = detect_face(rotated_image, face_detection)
                if face_bbox:
                    image = rotated_image
                    break

        if face_bbox:
            x1, y1, x2, y2 = face_bbox

            # Adjust padding to capture full head and neck
            padding_top = int(0.48 * (y2 - y1))
            padding_bottom = int(0.82 * (y2 - y1))
            padding_side = int(0.3 * (x2 - x1))

            # Apply padding
            h, w, _ = image.shape
            y1 = max(0, y1 - padding_top)
            y2 = min(h, y2 + padding_bottom)
            x1 = max(0, x1 - padding_side)
            x2 = min(w, x2 + padding_side)

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            # Save the cropped image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Processed and saved: {filename}")
        else:
            skipped_files.append((filename, "Face not detected"))
            no_face_path = os.path.join(no_face_folder, filename)
            shutil.move(input_path, no_face_path)
            print(f"No face detected after rotations. Moved to 'no_face' folder: {filename}")

    # Release resources
    face_detection.close()
    
    # Print skipped file summary
    if skipped_files:
        print("\n Skipped Files Report:")
        for fname, reason in skipped_files:
            print(f"{fname} --> {reason}")
    else:
        print("\n All files processed successfully!")

    print("Processing complete!")

# input/input/Aayush script/Photos new list
# Set folder paths
input_folder = "today"
output_folder = "D:/Photo-Cropping-App/New-14-April-Output"
no_face_folder = "D:/Photo-Cropping-App/No-Face-detected"

# Run the function
crop_passport_photos(input_folder, output_folder, no_face_folder)

