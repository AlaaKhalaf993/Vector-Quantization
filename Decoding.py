import json
import os
from PIL import Image


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_codebook(filename):
    """Load metadata and codebook vectors from a JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in codebook file: {error}") from error

    if type(data) is not dict:
        raise ValueError("Codebook JSON must contain an object.")

    required_fields = (
        "mode",
        "original_height",
        "original_width",
        "vector_height",
        "vector_width",
        "number_of_codebooks",
        "codebook",
    )
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(
            "Missing required codebook field(s): " + ", ".join(missing_fields)
        )

    mode = data["mode"]
    if type(mode) is not str or mode not in ("L", "RGB"):
        raise ValueError('Image mode must be "L" or "RGB".')

    original_height = _positive_integer(data["original_height"], "original_height")
    original_width = _positive_integer(data["original_width"], "original_width")
    vector_height = _positive_integer(data["vector_height"], "vector_height")
    vector_width = _positive_integer(data["vector_width"], "vector_width")
    number_of_codebooks = _positive_integer(
        data["number_of_codebooks"], "number_of_codebooks"
    )

    raw_codebook = data["codebook"]
    if type(raw_codebook) is not dict or not raw_codebook:
        raise ValueError("Codebook must be a non-empty JSON object.")

    codebook = {}
    for key, vector in raw_codebook.items():
        if type(key) is not str or not key.isascii() or not key.isdigit():
            raise ValueError("Codebook keys must be non-negative integer strings.")

        index = int(key)
        if str(index) != key:
            raise ValueError(f"Invalid codebook index key: {key!r}.")
        codebook[index] = vector

    if len(codebook) != number_of_codebooks:
        raise ValueError("number_of_codebooks does not match the codebook size.")
    if set(codebook) != set(range(number_of_codebooks)):
        raise ValueError("Codebook indices must be consecutive, starting from 0.")

    for index, vector in codebook.items():
        if type(vector) is not list or len(vector) != vector_height:
            raise ValueError(
                f"Codebook vector {index} must have {vector_height} rows."
            )

        for row in vector:
            if type(row) is not list or len(row) != vector_width:
                raise ValueError(
                    f"Each row in codebook vector {index} must have "
                    f"{vector_width} pixels."
                )

            for pixel in row:
                if mode == "L":
                    if type(pixel) is not int or not 0 <= pixel <= 255:
                        raise ValueError(
                            f"Invalid grayscale pixel in codebook vector {index}."
                        )
                else:
                    if type(pixel) is not list or len(pixel) != 3:
                        raise ValueError(
                            f"RGB pixels in codebook vector {index} "
                            "must have three channels."
                        )
                    if any(
                        type(channel) is not int or not 0 <= channel <= 255
                        for channel in pixel
                    ):
                        raise ValueError(
                            f"Invalid RGB channel in codebook vector {index}."
                        )

    metadata = {
        "mode": mode,
        "original_height": original_height,
        "original_width": original_width,
        "vector_height": vector_height,
        "vector_width": vector_width,
        "number_of_codebooks": number_of_codebooks,
    }
    return metadata, codebook


def load_binary_indices(filename, codebook_size, total_indices):
    """Read MSB-first indices from a byte stream padded with zero bits at the end."""
    if type(codebook_size) is not int or codebook_size <= 0:
        raise ValueError("Codebook size must be a positive integer.")
    if type(total_indices) is not int or total_indices < 0:
        raise ValueError("Expected index count must be a non-negative integer.")

    with open(filename, "rb") as file:
        payload = file.read()

    bits_per_index = (codebook_size - 1).bit_length()
    expected_bits = total_indices * bits_per_index
    expected_bytes = (expected_bits + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Binary file has {len(payload)} bytes; expected exactly "
            f"{expected_bytes} bytes for {total_indices} indices."
        )

    if codebook_size == 1:
        return [0] * total_indices

    padding_bits = expected_bytes * 8 - expected_bits
    if padding_bits:
        padding_mask = (1 << padding_bits) - 1
        if payload[-1] & padding_mask:
            raise ValueError("Trailing padding bits must all be zero.")

    indices = []
    current_index = 0
    bits_in_index = 0

    for bit_position in range(expected_bits):
        byte = payload[bit_position // 8]
        bit = (byte >> (7 - bit_position % 8)) & 1
        current_index = (current_index << 1) | bit
        bits_in_index += 1

        if bits_in_index == bits_per_index:
            if current_index >= codebook_size:
                raise ValueError(
                    f"Decoded index {current_index} is outside the codebook "
                    f"range 0..{codebook_size - 1}."
                )
            indices.append(current_index)
            current_index = 0
            bits_in_index = 0

    return indices


def reconstruct_index_matrix(flat_indices, block_rows, block_cols):
    expected_count = block_rows * block_cols
    if len(flat_indices) != expected_count:
        raise ValueError(
            "Decoded index count does not match the expected block dimensions."
        )
    return [
        flat_indices[row * block_cols:(row + 1) * block_cols]
        for row in range(block_rows)
    ]


# 4. Remove padding from the image
def remove_padding(image, original_height, original_width):

    return [
        row[:original_width]
        for row in image[:original_height]
    ]


# 5. Reconstruct the padded image from its index matrix.
def reconstruct_compressed_image(indices, codebook, vector_height):
    image = []

    for index_row in indices:
        vector_rows = [[] for _ in range(vector_height)]
        for index in index_row:
            vector = codebook[index]
            for row_number, vector_row in enumerate(vector):
                vector_rows[row_number].extend(vector_row)
        image.extend(vector_rows)

    return image


# 6. Save the reconstructed image and pixel matrix.
def save_image(image, mode):
    if mode not in ("L", "RGB"):
        raise ValueError('Image mode must be "L" or "RGB".')
    if not image or not image[0]:
        raise ValueError("Cannot save an empty image.")

    width = len(image[0])
    if any(len(row) != width for row in image):
        raise ValueError("Reconstructed image rows must have equal widths.")

    os.makedirs("output", exist_ok=True)
    height = len(image)

    with open(
        os.path.join("output", "reconstructed_image.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(image, file)

    if mode == "L":
        pixels = bytes(pixel for row in image for pixel in row)
    else:
        pixels = bytes(
            channel
            for row in image
            for pixel in row
            for channel in pixel
        )

    Image.frombytes(mode, (width, height), pixels).save(
        os.path.join("output", "reconstructed_image.png")
    )

    print("\nImage saved successfully!")


# 7. Run the decoder using a codebook JSON file and a binary index file.
def run_decoding():
    print("\n===== Vector Quantization Decoder =====")

    try:
        codebook_filename = input("\nEnter Codebook JSON filename: ").strip()
        binary_filename = input("Enter Compressed Image Binary filename: ").strip()
        if not codebook_filename or not binary_filename:
            raise ValueError("Both input filenames are required.")

        metadata, codebook = load_codebook(codebook_filename)
        original_height = metadata["original_height"]
        original_width = metadata["original_width"]
        vector_height = metadata["vector_height"]
        vector_width = metadata["vector_width"]

        block_rows = (original_height + vector_height - 1) // vector_height
        block_cols = (original_width + vector_width - 1) // vector_width
        total_indices = block_rows * block_cols

        flat_indices = load_binary_indices(
            binary_filename, len(codebook), total_indices
        )
        index_matrix = reconstruct_index_matrix(
            flat_indices, block_rows, block_cols
        )
        padded_image = reconstruct_compressed_image(
            index_matrix, codebook, vector_height
        )

        expected_height = block_rows * vector_height
        expected_width = block_cols * vector_width
        if len(padded_image) != expected_height or any(
            len(row) != expected_width for row in padded_image
        ):
            raise ValueError("Reconstructed padded image dimensions are inconsistent.")

        image = remove_padding(padded_image, original_height, original_width)
        if len(image) != original_height or any(
            len(row) != original_width for row in image
        ):
            raise ValueError("Reconstructed image does not match the original dimensions.")

        save_image(image, metadata["mode"])

        print("\nDecoding completed successfully!")
        print("Image mode:", metadata["mode"])
        print("Vector dimensions:", vector_height, "x", vector_width)
        print("Reconstructed image dimensions:", original_height, "x", original_width)
        print("\nReconstructed pixel matrix:")
        for row in image:
            print(row)

    except (
        EOFError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print("\nError:", error)


if __name__ == "__main__":
    run_decoding()

