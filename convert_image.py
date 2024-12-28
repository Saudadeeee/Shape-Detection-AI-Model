from PIL import Image
import numpy as np

def convert_image(input_path, output_path):
    # Open the image file
    img = Image.open(input_path)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 400x400
    img = img.resize((400, 400))
    
    # Convert to binary
    img_array = np.array(img)
    binary_img_array = (img_array > 127).astype(np.uint8)
    
    # Save the binary image data to a .bin file
    binary_img_array.tofile(output_path)

if __name__ == "__main__":
    input_path = "test.png"
    output_path = "test_binary.bin"
    convert_image(input_path, output_path)
    print(f"Converted image saved to {output_path}")
