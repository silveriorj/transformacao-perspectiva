import cv2
import numpy as np
import argparse

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


def draw_square(image, points):
    """
    Draws a square on the image based on the provided points.
    """
    for i in range(len(points)):
        cv2.line(image, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)
    return image


def draw_points(image, points):
    """
    Draws points on the image based on the provided points.
    """
    for point in points:
        cv2.circle(image, point, 5, (255, 0, 0), -1)
    return image


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


def transform_affine(image, src_points, dst_points, output_size):
    """
    Transforms the perspective of a flatten image using OpenCV.

    :param image: Input image as a NumPy array.
    :param src_points: List of three points (x, y) defining the source quadrilateral.
    :param output_size: Tuple (width, height) for the output image size.
    :return: Perspective-transformed image.
    """
    # Compute the perspective transformation matrix
    matrix = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    transformed_image = cv2.warpAffine(image, matrix, output_size)
    return transformed_image


def main(image_path, method):
    """
    Main function to handle image loading, point selection, and transformation.
    """
    global src_points

    # Load an image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    
    if method == "perspective":
        # Reset src_points for perspective transformation
        src_points = []
        # Create a window and set a mouse callback to select points
        cv2.imshow("Select Points", image)
        cv2.setMouseCallback("Select Points", select_points)

        print("Select 3 points on the image by clicking on them.")
        while len(src_points) < 3:
            # Draw the selected points on the image
            image_copy = image.copy()
            image_copy = draw_points(image_copy, src_points, )
            cv2.imshow("Select Points", image_copy)
            cv2.waitKey(1)

        cv2.destroyWindow("Select Points")

        # Define destination points and output size
        dst_points = [(0, 0), (400, 0), (400, 400)]
        output_size = (400, 400)

        # Perform perspective transformation
        transformed_image = transform_affine(image, src_points, dst_points, output_size)
    
    elif method == "flatten":
        # Reset src_points for flatten transformation
        src_points = []
        # Create a window and set a mouse callback to select points
        cv2.imshow("Select Points", image)
        cv2.setMouseCallback("Select Points", select_points)

        print("Select 4 points on the image by clicking on them.")
        while len(src_points) < 4:
            # Draw the selected points on the image
            image_copy = image.copy()
            image_copy = draw_square(image_copy, src_points)
            cv2.imshow("Select Points", image_copy)
            cv2.waitKey(1)

        cv2.destroyWindow("Select Points")

        # Define destination points and output size
        dst_points = [(0, 0), (400, 0), (400, 400), (0, 400)]
        output_size = (400, 400)

        # Perform flatten transformation
        transformed_image = transform_perspective(image, src_points, dst_points, output_size)

    else:
        print("Invalid method. Use 'flatten' or 'perspective'.")
        return

    # Save or display the transformed image
    cv2.imwrite("output.jpg", transformed_image)
    cv2.imshow("Transformed Image", transformed_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # selects the method: flatten or perspective
    parser = argparse.ArgumentParser(description="Select transformation method")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--method", choices=["flatten", "perspective"], default="flatten", help="Transformation method")
    args = parser.parse_args()

    main(args.image_path, args.method)