import json
import os
import ast
from PIL import Image


# 1. Load data from JSON files
def load_data():

    with open("codebook.json", "r") as file:
        codebook = json.load(file)

    with open("compressed_image.json", "r") as file:
        compressed_data = json.load(file)

    codebook = {
        int(key): value
        for key, value in codebook.items()
    }

    compressed_image = compressed_data["indices"]
    original_height = compressed_data["original_height"]
    original_width = compressed_data["original_width"]
    color_mode = compressed_data["mode"]

    return (
        codebook,
        compressed_image,
        original_height,
        original_width,
        color_mode
    )


# 2. Enter data manually
def manual_input():

    print("\nEnter Codebook:")
    codebook = ast.literal_eval(input())

    print("\nEnter Compressed Image:")
    compressed_image = ast.literal_eval(input())

    print("\nEnter Original Height:")
    original_height = int(input())

    print("\nEnter Original Width:")
    original_width = int(input())

    print("\nChoose Image Type:")
    print("1. Grayscale")
    print("2. RGB")

    choice = input("Choose an option: ")

    if choice == "1":
        color_mode = "L"
    elif choice == "2":
        color_mode = "RGB"
    else:
        raise ValueError("Invalid image type!")

    codebook = {
        int(key): value
        for key, value in codebook.items()
    }

    return codebook, compressed_image, original_height, original_width, color_mode


# 3. Validate input data
def validate_data(codebook, compressed_image, color_mode):

    if type(codebook) is not dict or not codebook:
        raise ValueError("Invalid or empty codebook!")

    if type(compressed_image) is not list or not compressed_image:
        raise ValueError("Invalid or empty compressed image!")

    if type(compressed_image[0]) is not list or not compressed_image[0]:
        raise ValueError("Invalid compressed image rows!")

    if color_mode not in ("L", "RGB"):
        raise ValueError("Invalid image mode!")

    row_size = len(compressed_image[0])

    for row in compressed_image:

        if type(row) is not list or len(row) != row_size:
            raise ValueError("Compressed image must be rectangular!")

    first_block = next(iter(codebook.values()))

    if type(first_block) is not list or not first_block:
        raise ValueError("Invalid codebook block!")

    if type(first_block[0]) is not list or not first_block[0]:
        raise ValueError("Invalid codebook block rows!")

    block_height = len(first_block)
    block_width = len(first_block[0])

    for block in codebook.values():

        if type(block) is not list or len(block) != block_height:
            raise ValueError("Inconsistent block height!")

        for row in block:

            if type(row) is not list or len(row) != block_width:
                raise ValueError("Inconsistent block width!")

            for pixel in row:

                if color_mode == "L":

                    if type(pixel) is not int or not 0 <= pixel <= 255:
                        raise ValueError("Invalid grayscale pixel!")

                elif color_mode == "RGB":

                    if type(pixel) is not list or len(pixel) != 3:
                        raise ValueError("Invalid RGB pixel!")

                    for channel in pixel:

                        if type(channel) is not int or not 0 <= channel <= 255:
                            raise ValueError("Invalid RGB channel!")

    for row in compressed_image:

        for index in row:

            if type(index) is not int:
                raise ValueError("Invalid codebook index!")

            if index not in codebook:
                raise ValueError(f"Index {index} not found in codebook!")

    return block_height, block_width


# 4. Remove padding from the image
def remove_padding(image, original_height, original_width):

    return [
        row[:original_width]
        for row in image[:original_height]
    ]


# 5. Reconstruct compressed image
def reconstruct_compressed_image(compressed_image, codebook):

    first_block = next(iter(codebook.values()))
    block_height = len(first_block)
    reconstructed_image = []

    for indices in compressed_image:
        block_rows = [[] for _ in range(block_height)]
        for index in indices:
            block = codebook[index]

            for row_index, block_row_pixels in enumerate(block):
                block_rows[row_index].extend(block_row_pixels)

        reconstructed_image.extend(block_rows)
    return reconstructed_image


# 6. Save reconstructed image
def save_image(reconstructed_image, color_mode):

    if not reconstructed_image or not reconstructed_image[0]:
        raise ValueError("Cannot save an empty image!")

    width = len(reconstructed_image[0])

    for row in reconstructed_image:
        if len(row) != width:
            raise ValueError("Image rows must have the same width!")

    os.makedirs("output", exist_ok=True)

    height = len(reconstructed_image)

    with open("output/reconstructed_image.json", "w") as file:
        json.dump(reconstructed_image, file)

    if color_mode == "L":
        pixels = bytes(pixel for row in reconstructed_image for pixel in row)
    else:
        pixels = bytes(
            channel
            for row in reconstructed_image
            for pixel in row
            for channel in pixel
        )

    image_mode = "L" if color_mode == "L" else "RGB"
    image = Image.frombytes(image_mode, (width, height), pixels)
    image.save("output/reconstructed_image.png")

    print("\nImage saved successfully!")


# 7. Run decoding process
def run_decoding():

    print("\n===== Decoding Mode =====")
    print("1. Enter Data Manually")
    print("2. Load Existing Data")

    choice = input("Choose an option: ")

    try:

        if choice == "1":
            codebook, compressed_image, original_height, original_width, color_mode = manual_input()

        elif choice == "2":
            codebook, compressed_image, original_height, original_width, color_mode = load_data()

        else:
            print("Invalid choice!")
            return

        block_height, block_width = validate_data(
            codebook, compressed_image, color_mode
        )

        reconstructed_image = reconstruct_compressed_image(
            compressed_image, codebook
        )

        # Validate original dimensions
        if type(original_height) is not int or type(original_width) is not int:
            raise ValueError("Original dimensions must be integers!")

        if original_height <= 0 or original_width <= 0:
            raise ValueError("Original dimensions must be positive!")

        if (original_height - 1) // block_height != (
            len(reconstructed_image) - 1
        ) // block_height:
            raise ValueError("Invalid original height!")

        if (original_width - 1) // block_width != (
            len(reconstructed_image[0]) - 1
        ) // block_width:
            raise ValueError("Invalid original width!")

        # Remove padding
        reconstructed_image = remove_padding(
            reconstructed_image,
            original_height,
            original_width
        )

        save_image(reconstructed_image, color_mode)

        print("\nDecoding completed successfully!")

        print("Image Mode:", color_mode)
        print("Block size:", block_height, "x", block_width)

        print(
            "Reconstructed image dimensions:",
            len(reconstructed_image),
            "x",
            len(reconstructed_image[0])
        )

        print("\nReconstructed Image:")

        for row in reconstructed_image:
            print(row)

    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        json.JSONDecodeError,
        SyntaxError,
        TypeError,
        IndexError
    ) as error:

        print("\nError:", error)

