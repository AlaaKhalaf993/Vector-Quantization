
import numpy as np
from PIL import Image
import struct
import os


def compress():

    image_path = input("Image path: ")
    block_size = int(input("Block size: "))
    codebook_size = int(input("Number of codebook vectors: "))

    image = np.array(Image.open(image_path).convert("RGB"))
    original_height, original_width, channels = image.shape

    # Padding
    padded_height = ((original_height + block_size - 1) // block_size) * block_size
    padded_width = ((original_width + block_size - 1) // block_size) * block_size

    padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded_image[:original_height, :original_width] = image

    # Divide image into vectors
    vectors = []

    for row in range(0, padded_height, block_size):
        for column in range(0, padded_width, block_size):

            block = padded_image[row:row+block_size, column:column+block_size]
            vectors.append(block.flatten())

    vectors = np.array(vectors, dtype=float)

    if codebook_size < 1 or codebook_size > len(vectors):
        print("Invalid codebook size")
        return

    # Create initial codebook
    codebook = []

    for index in range(codebook_size):
        codebook.append(vectors[index].copy())

    codebook = np.array(codebook)

    # Improve codebook using K-means
    for iteration in range(20):

        groups = []

        for index in range(codebook_size):
            groups.append([])

        for vector in vectors:

            distances = []

            for center in codebook:
                distance = np.sum((vector - center) ** 2)
                distances.append(distance)

            nearest_index = distances.index(min(distances))
            groups[nearest_index].append(vector)

        for index in range(codebook_size):

            if len(groups[index]) > 0:
                codebook[index] = np.mean(groups[index], axis=0)

    # Find nearest codebook vector for every image vector
    compressed_indexes = []

    for vector in vectors:

        distances = []

        for center in codebook:
            distance = np.sum((vector - center) ** 2)
            distances.append(distance)

        nearest_index = distances.index(min(distances))
        compressed_indexes.append(nearest_index)

    # Save codebook and image information
    with open("codebook.bin", "wb") as file:

        file.write(struct.pack(
            "<5I",
            original_height,
            original_width,
            block_size,
            codebook_size,
            len(compressed_indexes)
        ))

        codebook.astype(np.float32).tofile(file)

    # Save compressed binary stream
    if codebook_size <= 256:
        index_type = np.uint8
    elif codebook_size <= 65536:
        index_type = np.uint16
    else:
        index_type = np.uint32

    np.array(compressed_indexes, dtype=index_type).tofile("compressed.bin")

    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize("codebook.bin") + os.path.getsize("compressed.bin")

    print("\nCompression completed!")
    print("Original size:", original_size, "bytes")
    print("Compressed size:", compressed_size, "bytes")
    print("Compression ratio:", round(original_size / compressed_size, 2))


def decompress():

    # Read codebook
    with open("codebook.bin", "rb") as file:

        original_height, original_width, block_size, codebook_size, vector_count = struct.unpack(
            "<5I", file.read(20)
        )

        codebook = np.fromfile(file, dtype=np.float32)
        codebook = codebook.reshape(codebook_size, block_size * block_size * 3)

    # Read compressed indexes
    if codebook_size <= 256:
        index_type = np.uint8
    elif codebook_size <= 65536:
        index_type = np.uint16
    else:
        index_type = np.uint32

    compressed_indexes = np.fromfile("compressed.bin", dtype=index_type)

    # Reconstruct padded image
    padded_height = ((original_height + block_size - 1) // block_size) * block_size
    padded_width = ((original_width + block_size - 1) // block_size) * block_size

    reconstructed_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)

    vector_index = 0

    for row in range(0, padded_height, block_size):
        for column in range(0, padded_width, block_size):

            codebook_index = compressed_indexes[vector_index]

            vector = codebook[codebook_index]
            block = vector.reshape(block_size, block_size, 3)

            reconstructed_image[row:row+block_size, column:column+block_size] = (
                block.clip(0, 255).astype(np.uint8)
            )

            vector_index += 1

    # Remove padding
    reconstructed_image = reconstructed_image[:original_height, :original_width]

    Image.fromarray(reconstructed_image).save("decompressed.png")

    print("Decompression completed!")
    print("Saved as decompressed.png")


# Main menu
while True:

    print("\n1. Compression")
    print("2. Decompression")
    print("3. Exit")

    choice = input("Choose: ")

    if choice == "1":
        compress()

    elif choice == "2":
        decompress()

    elif choice == "3":
        break

    else:
        print("Invalid choice")
