
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

    return codebook, compressed_image, original_height, original_width

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

    codebook = {
        int(key): value
        for key, value in codebook.items()
    }

    return codebook, compressed_image, original_height, original_width


# 3. Validate input data
def validate_data(codebook, compressed_image):

    if not isinstance(codebook, dict) or not codebook:
        raise ValueError("Invalid or empty codebook!")

    if not isinstance(compressed_image, list) or not compressed_image:
        raise ValueError("Invalid or empty compressed image!")

    if not isinstance(compressed_image[0], list) or not compressed_image[0]:
        raise ValueError("Invalid compressed image rows!")

    row_size = len(compressed_image[0])

    for row in compressed_image:

        if not isinstance(row, list) or len(row) != row_size:
            raise ValueError("Compressed image must be rectangular!")

    first_block = next(iter(codebook.values()))

    if not isinstance(first_block, list) or not first_block:
        raise ValueError("Invalid codebook block!")

    if not isinstance(first_block[0], list) or not first_block[0]:
        raise ValueError("Invalid codebook block rows!")

    block_height = len(first_block)
    block_width = len(first_block[0])

    for block in codebook.values():

        if not isinstance(block, list) or len(block) != block_height:
            raise ValueError("Inconsistent block height!")

        for row in block:

            if not isinstance(row, list) or len(row) != block_width:
                raise ValueError("Inconsistent block width!")

            for pixel in row:

                if not isinstance(pixel, (int, float)):
                    raise ValueError("Invalid pixel value!")

                if not 0 <= pixel <= 255:
                    raise ValueError("Pixel values must be between 0 and 255!")

    for row in compressed_image:

        for index in row:

            if not isinstance(index, int):
                raise ValueError("Invalid codebook index!")

            if index not in codebook:
                raise ValueError(f"Index {index} not found in codebook!")

    return block_height, block_width


def remove_padding(image, original_height, original_width):

    return [
        row[:original_width]
        for row in image[:original_height]
    ]


# 4. Reconstruct compressed image
def reconstruct_compressed_image(compressed_image, codebook):

    block_height = len(next(iter(codebook.values())))
    block_width = len(next(iter(codebook.values()))[0])

    block_row_count = len(compressed_image)
    block_count = len(compressed_image[0])

    height = block_row_count * block_height
    width = block_count * block_width

    reconstructed_image = [[0] * width for _ in range(height)]

    for block_row_index, indices in enumerate(compressed_image):

        row_offset = block_row_index * block_height
        column_offset = 0

        for index in indices:

            block = codebook[index]

            for row_index, block_row_pixels in enumerate(block):

                target_row = reconstructed_image[row_offset + row_index]

                target_row[column_offset:column_offset + block_width] = block_row_pixels

            column_offset += block_width

    return reconstructed_image


# 5. Save reconstructed image
def save_image(reconstructed_image):

    os.makedirs("output", exist_ok=True)

    height = len(reconstructed_image)
    width = len(reconstructed_image[0])

    with open("output/reconstructed_image.json", "w") as file:
        json.dump(reconstructed_image, file)

    pixels = bytes(
        int(pixel)
        for row in reconstructed_image
        for pixel in row
    )

    image = Image.frombytes("L", (width, height), pixels)

    image.save("output/reconstructed_image.png")

    print("\nImage saved successfully!")


# 6. Run decoding process

def run_decoding():

    print("\n===== Decoding Mode =====")
    print("1. Enter Data Manually")
    print("2. Load Existing Data")

    choice = input("Choose an option: ")

    try:

        if choice == "1":
            codebook, compressed_image, original_height, original_width = manual_input()

        elif choice == "2":
            codebook, compressed_image, original_height, original_width = load_data()

        else:
            print("Invalid choice!")
            return

        block_height, block_width = validate_data(
            codebook, compressed_image
        )

        reconstructed_image = reconstruct_compressed_image(
            compressed_image, codebook
        )

       # Validate original dimensions
        if not isinstance(original_height, int) or not isinstance(original_width, int):
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

        save_image(reconstructed_image)

        print("\nDecoding completed successfully!")

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

