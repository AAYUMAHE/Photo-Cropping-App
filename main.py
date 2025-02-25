import cv2
import mediapipe as mp
import os
import shutil


def crop_passport_photos(input_folder, output_folder, no_face_folder):
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(no_face_folder, exist_ok=True)

    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue

        # Convert image to RGB for Mediapipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_image)
        if results.detections:
            for detection in results.detections:
                # Extract bounding box for the first face detected
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)

                # Adjust padding to capture full head and neck
                padding_top = int(0.48 * (y2 - y1))  # Increased padding for head
                padding_bottom = int(0.4 * (y2 - y1))  # Keep neck padding the same
                padding_side = int(0.2 * (x2 - x1))  # Include shoulders

                # Apply padding to the bounding box
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
                break
        else:
            # If no face is detected, move the image to the 'no_face' folder
            no_face_path = os.path.join(no_face_folder, filename)
            shutil.move(input_path, no_face_path)
            print(f"No face detected. Moved to 'no_face' folder: {filename}")

    # Release the Mediapipe resources
    face_detection.close()
    print("Processing complete!")

# Please put the copied path of folders , only.
input_folder = "D:/Photo-Cropping-App/photo_cropping/photo_crop"
output_folder = "D:/Photo-Cropping-App/New-25-Feb-Output"
no_face_folder = "D:/Photo-Cropping-App/No-Face-detected"
crop_passport_photos(input_folder, output_folder, no_face_folder)
