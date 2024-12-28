from PIL import Image
import numpy as np

def convert_image(input_path, output_path):
    img = Image.open(input_path)

    img = img.convert('L')
    
    img = img.resize((400, 400))
    

    img_array = np.array(img)
    binary_img_array = (img_array > 127).astype(np.uint8)

    binary_img_array.tofile(output_path)

if __name__ == "__main__":
    input_path = "test.png"
    output_path = "test_binary.bin"
    convert_image(input_path, output_path)
    print(f"Converted image saved to {output_path}")
