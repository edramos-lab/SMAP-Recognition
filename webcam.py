import cv2

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()

        # Display the frame in a window
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()