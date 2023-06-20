import cv2
import numpy as np

def segment_image(image_path, sigma=0.8, k=300, min_size=100):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the graph-based image segmentation
    segmenter = cv2.ximgproc.createSuperpixelSLIC(image, algorithm=cv2.ximgproc.SLIC, region_size=k, ruler=10.0)
    segmenter.iterate()
    segments = segmenter.getLabels()
    
    # Generate a random color for each segment
    num_segments = np.max(segments) + 1
    colors = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)
    
    # Create a segmented output image
    segmented_image = np.zeros_like(image)
    for segment_id in np.unique(segments):
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[segments == segment_id] = 255
        color = colors[segment_id]
        color = tuple(map(int, color))  # Convert color array to integers
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(segmented_image, contours, -1, color, thickness=cv2.FILLED)
    
    return segmented_image

# Test the image segmentation algorithm
image_path = "C:/Users/arunk/OneDrive/Desktop/b/vector-illustration-of-a-tiger-on-red-background-KK149H.jpg"
segmented_image = segment_image(image_path, sigma=0.8, k=300, min_size=100)

if segmented_image is not None:
    # Display the original image and segmented image
    cv2.imshow('Original Image', cv2.imread(image_path))
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
