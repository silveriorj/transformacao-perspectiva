import cv2
import numpy as np
import sys

# List to store points
src_points = []

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to select points on the image.
    """
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN and len(src_points) < 4:
        src_points.append((x, y))
        print(f"Point selected: {x, y}")


def transform_perspective(image, src_points, dst_points, output_size):
    """
    Transforms the perspective of an image using OpenCV.

    :param image: Input image as a NumPy array.
    :param src_points: List of four points (x, y) defining the source quadrilateral.
    :param dst_points: List of four points (x, y) defining the destination quadrilateral.
    :param output_size: Tuple (width, height) for the output image size.
    :return: Perspective-transformed image.
    """
    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    
    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, matrix, output_size)
    
    return transformed_image


def main(image_path):
    global src_points

    # Load an image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # Create a window and set a mouse callback to select points
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", select_points)

    print("Select 4 points on the image by clicking on them.")
    while len(src_points) < 4:
        cv2.imshow("Select Points", image)
        cv2.waitKey(1)

    cv2.destroyWindow("Select Points")

    # Define destination points and output size
    dst_points = [(0, 0), (400, 0), (400, 400), (0, 400)]
    output_size = (400, 400)

    # Perform perspective transformation
    transformed_image = transform_perspective(image, src_points, dst_points, output_size)

    # Save or display the transformed image
    cv2.imwrite("output.jpg", transformed_image)
    cv2.imshow("Transformed Image", transformed_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
    else:
        main(sys.argv[1])