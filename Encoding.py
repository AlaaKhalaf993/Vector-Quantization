from PIL import Image
import numpy as np
import math
import json
import os
from CompressionRatio import calculate_compression_ratio


def read_image(name):
    with Image.open(name) as img:
        if img.mode not in ("L", "RGB"):
            raise ValueError('Input image mode must be "L" or "RGB".')
        return img.mode, np.array(img)

def pad_image(img_array, block_height, block_width):
    height, width = img_array.shape[:2]
    padding_width = (block_width - (width % block_width)) % block_width
    padding_height = (block_height - (height % block_height)) % block_height

    if img_array.ndim == 2:
        pad_width = ((0, padding_height), (0, padding_width))
    else:
        pad_width = ((0, padding_height), (0, padding_width), (0, 0))
    padded_img = np.pad(img_array, pad_width=pad_width, mode='edge')
    
    return padded_img, (height, width)

def extract_vectors(image, block_height, block_width):
    height, width = image.shape[:2]
    vectors = []
    for x in range(0, height, block_height):
        for y in range(0, width, block_width):
            vector = image[x : x + block_height, y : y + block_width]
            vectors.append(vector.flatten())
    return np.array(vectors, dtype=float)
#----------------
#code book generation
def calc_distance(vector1, vector2):
    distance = np.sum(np.abs(np.subtract(np.array(vector1), np.array(vector2))))

    return distance

def split_codebook(codebook):
    new_codebook=[]
    centroid1 = []
    centroid2 = []
    for centroid in codebook:
        centroid1 = [math.ceil(elem - 1) for elem in centroid]
        centroid2 = [math.floor(elem + 1) for elem in centroid]
        new_codebook.append(centroid1)
        new_codebook.append(centroid2)

    codebook = new_codebook
    return codebook


def assign_vectors(vectors, codebook):
    groups = [[] for _ in range(len(codebook))]
    assignments = []
    for vector in vectors:
        distances = []

        for centroid in codebook:
            distances.append(calc_distance(vector, centroid))
        
        nearest_group = np.argmin(distances)

        groups[nearest_group].append(vector)
        assignments.append(nearest_group)

    return groups, np.array(assignments)
        


def update_centroids(groups, codebook):
    newcodebook = []
    for i, group in enumerate(groups):
        if len(group) == 0:
            newcodebook.append(codebook[i])
        else:
            average = np.mean(group, axis=0)
            newcodebook.append(average)
    
    return np.array(newcodebook, dtype=float)

def generate_codebook(vectors, K):
    if type(K) is not int or K < 1:
        raise ValueError("K must be a positive integer.")

    codebook = np.mean(vectors, axis = 0, keepdims = True)
    while len(codebook) < K:
        codebook = split_codebook(codebook)[:K]

        groups, assignments = assign_vectors(vectors, codebook)

        codebook = update_centroids(groups, codebook)

    old_assignments = None

    for _ in range(100):
        groups, assignments = assign_vectors(vectors, codebook)
        new_codebook = update_centroids(groups, codebook)

        if (old_assignments is not None and np.array_equal(old_assignments, assignments)):
            codebook = new_codebook
            break

        old_assignments = assignments.copy()
        codebook = new_codebook

    else:
        raise RuntimeError("LBG failed to converge.")
    return codebook
#-------------------------
#encoding

def write_vector_bitstream(file, indices, K):

    bits_per_index = (K - 1).bit_length()
    if bits_per_index == 0:
        return

    bit_stream = ""
    for index in indices:
        if not 0 <= int(index) < K:
            raise ValueError("invalid codebook index")

        bit_stream += format(int(index), f'0{bits_per_index}b')

    padding = (8 - len(bit_stream) % 8) % 8

    bit_stream += "0" * padding

    if bit_stream:
        packed_bytes = int(bit_stream, 2).to_bytes(len(bit_stream) // 8, "big")
        file.write(packed_bytes)

def write_codebook(filename, codebook, original_height,
                        original_width, vector_height, vector_width, mode="L"):

    if mode not in ("L", "RGB"):
        raise ValueError('Image mode must be "L" or "RGB".')

    codebook = np.asarray(codebook, dtype=float)

    K = len(codebook)

    codebook = np.clip(np.rint(codebook), 0, 255).astype(np.uint8)
    if mode == "RGB":
        codebook = codebook.reshape(K, vector_height, vector_width, 3)
    else:
        codebook = codebook.reshape(K, vector_height, vector_width)

    data = {
        "mode": mode,
        "original_height": int(original_height),
        "original_width": int(original_width),
        "vector_height": int(vector_height),
        "vector_width": int(vector_width),
        "number_of_codebooks": K,
        "codebook": {
            str(i): vector.tolist()
            for i, vector in enumerate(codebook)
        }
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def compress(img, blockHeight, blockWidth, K, codebookfilename, comprfilename):
    if blockHeight <= 0 or blockWidth <= 0:
        raise ValueError("block dimensions must be positive")
    if type(K) is not int or K < 1:
        raise ValueError("K must be a positive integer.")

    mode, img_array = read_image(img)
    orgHeight, orgWidth = img_array.shape[:2]
    paddedimg, _ = pad_image(img_array, blockHeight, blockWidth)
    vectors = extract_vectors(paddedimg, blockHeight, blockWidth)
    codebook = generate_codebook(vectors, K)

    codebook = np.clip(np.rint(codebook), 0, 255).astype(np.uint8)
    write_codebook(
        codebookfilename,
        codebook,
        orgHeight,
        orgWidth,
        blockHeight,
        blockWidth,
        mode=mode,
    )
    
    _, indices = assign_vectors(vectors, codebook)
    with open(comprfilename, "wb") as file:
        write_vector_bitstream(file, indices, K)

    return calculate_compression_ratio(img, codebookfilename, comprfilename)



def run_encoding():

    print("===== Vector Quantization Compression =====")

    try:
        image_path = input("Enter image filename: ").strip()

        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image file not found.")

        block_height = int(input("Enter block height: "))
        block_width = int(input("Enter block width: "))

        if block_height <= 0 or block_width <= 0:
            raise ValueError("block dimensions must be positive")

        K = int(input("enter number of codebook vectors (K): "))

        if K < 1:
            raise ValueError("K must be a positive integer.")

        mode, img = read_image(image_path)

        height, width = img.shape[:2]

        padded_height = ((height + block_height - 1) // block_height) * block_height

        padded_width = ((width + block_width - 1) // block_width) * block_width

        number_of_blocks = ((padded_height // block_height) * (padded_width // block_width))

        codebook_filename = "codebook.json"
        compressed_filename = "compressed_image.bin"

        compression_ratio = compress(
            image_path,
            block_height,
            block_width,
            K,
            codebook_filename,
            compressed_filename,
        )

        if not os.path.isfile(codebook_filename):
            raise RuntimeError("Codebook file was not created.")

        if not os.path.isfile(compressed_filename):
            raise RuntimeError("Compressed image file was not created.")

        print("\n===== Compression Completed =====")

        print("Original image dimensions:", height, "x", width)
        print("Padded image dimensions:", padded_height, "x", padded_width)
        print("Block dimensions:", block_height, "x", block_width)
        print("Number of blocks:", number_of_blocks)
        print("Codebook size:", K)
        print("Image mode:", mode)

        print("\nGenerated files:")

        print(codebook_filename, "-", os.path.getsize(codebook_filename), "bytes")

        print(compressed_filename, "-", os.path.getsize(compressed_filename), "bytes")
        print("Compression ratio:", f"{compression_ratio:.2f}:1")

    except(ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        print("\nCompression Error:", error)


if __name__ == "__main__":
    run_encoding()