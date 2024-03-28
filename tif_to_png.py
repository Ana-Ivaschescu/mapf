from PIL import Image
import os

def tiff_to_png(input_path, output_path):
    # Open the TIFF file
    tiff_image = Image.open(input_path)

    # Generate the output PNG path with the same name
    output_png_path = os.path.splitext(output_path)[0] + ".png"

    # Convert and save as PNG
    tiff_image.save(output_png_path, format='PNG')

if __name__ == "__main__":
    # Specify the input directory containing TIFF files
    input_directory = "./data/preprocessed/2000/labels/change/D35/35-2012-0325-6765-LA93-0M50-E080"

    # Specify the output directory for PNG files
    output_directory = "output_directory"

    # Check if the input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Loop through each file in the input directory
        for filename in os.listdir(input_directory):
            if filename.endswith(".tiff") or filename.endswith(".tif"):
                input_tiff_path = os.path.join(input_directory, filename)
                output_png_path = os.path.join(output_directory, filename)

                # Convert TIFF to PNG
                tiff_to_png(input_tiff_path, output_png_path)

                print(f"Conversion complete. PNG file saved at '{output_png_path}'.")