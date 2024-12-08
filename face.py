import cv2
import dlib
import numpy as np

class EmotionDetector:
    def __init__(self):
        """
        Initialize facial landmark detector and emotion analysis tools
        """
        # Load pre-trained face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Load facial landmark predictor (ensure you've downloaded the landmark predictor file)
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def calculate_facial_landmarks(self, landmarks):
        """
        Extract key facial landmark coordinates for detailed analysis
        """
        # Mouth landmarks
        mouth_left = np.array([landmarks.part(48).x, landmarks.part(48).y])
        mouth_right = np.array([landmarks.part(54).x, landmarks.part(54).y])
        mouth_top = np.array([landmarks.part(51).x, landmarks.part(51).y])
        mouth_bottom = np.array([landmarks.part(57).x, landmarks.part(57).y])

        # Eye landmarks
        left_eye_top = np.array([landmarks.part(37).x, landmarks.part(37).y])
        left_eye_bottom = np.array([landmarks.part(41).x, landmarks.part(41).y])
        right_eye_top = np.array([landmarks.part(44).x, landmarks.part(44).y])
        right_eye_bottom = np.array([landmarks.part(46).x, landmarks.part(46).y])

        # Eyebrow landmarks
        left_eyebrow_inner = np.array([landmarks.part(17).x, landmarks.part(17).y])
        left_eyebrow_outer = np.array([landmarks.part(21).x, landmarks.part(21).y])
        right_eyebrow_inner = np.array([landmarks.part(22).x, landmarks.part(22).y])
        right_eyebrow_outer = np.array([landmarks.part(26).x, landmarks.part(26).y])

        return {
            'mouth_width': np.linalg.norm(mouth_left - mouth_right),
            'mouth_height': np.linalg.norm(mouth_top - mouth_bottom),
            'left_eye_openness': np.linalg.norm(left_eye_top - left_eye_bottom),
            'right_eye_openness': np.linalg.norm(right_eye_top - right_eye_bottom),
            'left_eyebrow_slope': (left_eyebrow_outer[1] - left_eyebrow_inner[1]) / 
                                   (left_eyebrow_outer[0] - left_eyebrow_inner[0] + 1e-6),
            'right_eyebrow_slope': (right_eyebrow_outer[1] - right_eyebrow_inner[1]) / 
                                    (right_eyebrow_outer[0] - right_eyebrow_inner[0] + 1e-6)
        }

    def detect_emotion(self, frame):
        """
        Detect emotions using facial landmark analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        for face in faces:
            # Detect landmarks
            landmarks = self.predictor(gray, face)
            
            # Calculate detailed facial measurements
            facial_metrics = self.calculate_facial_landmarks(landmarks)
            
            # Emotion detection logic with detailed metrics
            emotion = "Neutral"
            debug_info = []

            # Happy detection (wide smile, raised mouth corners)
            mouth_smile_ratio = facial_metrics['mouth_width'] / facial_metrics['mouth_height']
            if mouth_smile_ratio > 3.5:
                emotion = "Happy"
                debug_info.append(f"Smile Ratio: {mouth_smile_ratio:.2f}")

            # Sad detection (downturned mouth, flattened eyebrows)
            elif (facial_metrics['left_eyebrow_slope'] > 0.2 and 
                  facial_metrics['right_eyebrow_slope'] > 0.2):
                emotion = "Sad"
                debug_info.append("Downturned Eyebrows")

            # Angry detection (furrowed eyebrows, tight mouth)
            elif (facial_metrics['left_eyebrow_slope'] < -0.3 and 
                  facial_metrics['right_eyebrow_slope'] < -0.3):
                emotion = "Angry"
                debug_info.append("Furrowed Eyebrows")

            # Surprise detection (wide eyes, raised eyebrows)
            elif (facial_metrics['left_eye_openness'] > 15 and 
                  facial_metrics['right_eye_openness'] > 15 and 
                  facial_metrics['left_eyebrow_slope'] < -0.5):
                emotion = "Surprised"
                debug_info.append("Wide Eyes & Raised Brows")

            # Bounding box
            x, y, w, h = (
                face.left(), face.top(), 
                face.width(), face.height()
            )
            
            # Draw rectangle and emotion text
            color = (0, 255, 0)  # Green for neutral and happy emotions
            if emotion == "Sad":
                color = (255, 0, 0)  # Blue for sad emotion
            elif emotion == "Angry":
                color = (0, 0, 255)  # Red for angry emotion
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion and debug info
            display_text = emotion
            if debug_info:
                display_text += f" ({', '.join(debug_info)})"
            
            cv2.putText(frame, display_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        color, 2)
        
        return frame

    def real_time_emotion_detection(self):
        """
        Perform real-time emotion detection using webcam
        """
        cap = cv2.VideoCapture(1)  # Use 1 or other indices if you have multiple cameras
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and classify emotions
            frame_with_emotions = self.detect_emotion(frame)
            
            # Display the frame
            cv2.imshow('Emotion Recognition', frame_with_emotions)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize emotion detector
    emotion_detector = EmotionDetector()
    
    # Start real-time emotion detection
    emotion_detector.real_time_emotion_detection()

if __name__ == "__main__":
    main()
