import cv2

# Load the pre-trained face cascade from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Get the screen dimensions
_, frame = video_capture.read()
screen_height, screen_width, _ = frame.shape

# Calculate the center coordinates of the screen
center_x = int(screen_width / 2)
center_y = int(screen_height / 2)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around detected faces and display the dot and line
    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the coordinates of the center of the bounding box
        face_center_x = x + int(w / 2)
        face_center_y = y + int(h / 2)
        
        # Draw a small green dot at the center of the screen
        cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)
        
        # Draw a thin green line from the dot to the center of the face bounding box
        cv2.line(frame, (center_x, center_y), (face_center_x, face_center_y), (0, 255, 0), 3)
    
    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
