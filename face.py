import cv2
import face_recognition

def detect_faces(frame):
    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    return face_locations

def encode_faces(images):
    encoded_faces = []
    
    for image in images:
        # Convert the image from BGR color (OpenCV default) to RGB color
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode the face
        face_encodings = face_recognition.face_encodings(rgb_image)[0]
        
        encoded_faces.append(face_encodings)
    
    return encoded_faces

def compare_faces(known_faces, unknown_face):
    # Compare the unknown face with known faces
    results = face_recognition.compare_faces(known_faces, unknown_face)
    
    return results

def main():
    # Load known face images and encode them
    known_images = ['known_face_1.jpg', 'known_face_2.jpg']  # Add more known face images if needed
    known_faces = encode_faces([cv2.imread(image) for image in known_images])
    
    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)
    
    # Initialize variables
    face_detected = False
    signed_in = False
    
    while True:
        # Read a single frame from the video capture
        ret, frame = video_capture.read()
        
        # Detect faces in the frame
        face_locations = detect_faces(frame)
        
        # If a face is detected and not already signed in, compare it with known faces
        if face_locations and not signed_in:
            # Extract the face encodings from the first detected face
            face_encodings = face_recognition.face_encodings(frame, face_locations)[0]
            
            # Compare the face with known faces
            results = compare_faces(known_faces, face_encodings)
            
            # If any known face is detected, sign in
            if True in results:
                signed_in = True
                print("Sign-in successful!")
        
        # Draw rectangles around the detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Show the resulting frame
        cv2.imshow('Face Sign-in', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
