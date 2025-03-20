import cv2
import os
import time

def collect_images(label, save_path, num_samples=200):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        cv2.imshow("Collecting Images", frame)
        filename = os.path.join(save_path, f"{label}_{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved: {filename}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    base_dir = "dataset"
    os.makedirs(base_dir, exist_ok=True)
    
    # Collect "without mask" images
    label = "without_mask"
    save_path = os.path.join(base_dir, label)
    os.makedirs(save_path, exist_ok=True)
    print("Starting collection for WITHOUT mask...")
    collect_images(label, save_path)
    
    # Wait for 5 seconds before switching
    print("Switch to WITH mask and wait 5 seconds...")
    time.sleep(5)
    
    # Collect "with mask" images
    label = "with_mask"
    save_path = os.path.join(base_dir, label)
    os.makedirs(save_path, exist_ok=True)
    print("Starting collection for WITH mask...")
    collect_images(label, save_path)
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()
