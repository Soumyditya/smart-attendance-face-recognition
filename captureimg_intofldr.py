import cv2
import os

# Camera setup
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Ask for person name
inp = input("Enter person name: ")

# Create the known_faces folder if missing
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

print("\n[INFO] Press 'c' to capture image, 'q' to quit without saving.\n")

while True:
    result, image = cam.read()
    if not result:
        print("[ERROR] No image detected. Please try again.")
        break

    cv2.imshow("Capture Face - " + inp, image)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # File path where image will be saved
        file_path = os.path.join(KNOWN_FACES_DIR, f"{inp}.png")
        cv2.imwrite(file_path, image)
        print(f"[INFO] Image saved successfully as {file_path}")
        break

    elif key == ord('q'):
        print("[INFO] Exiting without saving.")
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
