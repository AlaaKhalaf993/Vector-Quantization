from PIL import Image
import numpy as np
import math
import json
import os

#compression ratio function
def getRatio(codebookfilename, comprfilename):

    with open(codebookfilename, "r", encoding="utf-8") as file:
        data = json.load(file)

    height = data["original_height"]
    width = data["original_width"]
    vector_height = data["vector_height"]
    vector_width = data["vector_width"]
    K = data["number_of_codebooks"]
    mode = data["mode"]

    if mode == "L":
        bits_per_pixel = 8
    elif mode == "RGB":
        bits_per_pixel = 24
    else:
        raise ValueError("Unsupported image mode.")

    original_bits = height * width * bits_per_pixel

    block_rows = (height + vector_height - 1) // vector_height
    block_cols = (width + vector_width - 1) // vector_width

    number_of_blocks = block_rows * block_cols

    bits_per_index = (K - 1).bit_length()

    labels_bits = number_of_blocks * bits_per_index

    codebook_bits = (K * vector_height * vector_width * bits_per_pixel)

    compressed_bits = codebook_bits + labels_bits

    ratio = original_bits / compressed_bits

    change = ((original_bits - compressed_bits) / original_bits) * 100

    direction = "decreased" if change >= 0 else "increased"
    emoji = "(:" if change >= 0 else "):"

    print("\n─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")
    print("Vector Quantization - Compression Ratio")
    print("─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")

    print(f"\n.⋆❀° Original image size: {height} x {width} x {bits_per_pixel} = {original_bits} bits")

    print(f".⋆❀° Number of blocks: {block_rows} x {block_cols} = {number_of_blocks}")

    print(f".⋆❀° Bits per label: {bits_per_index} bits")

    print(f".⋆❀° Labels size: {number_of_blocks} x {bits_per_index} = {labels_bits} bits")

    print(f".⋆❀° Codebook size: {K} x {vector_height} x {vector_width} x {bits_per_pixel} = {codebook_bits} bits")

    print(f".⋆❀° Total compressed size: {codebook_bits} + {labels_bits} = {compressed_bits} bits")

    print(f"\n.⋆❀° Compression Ratio = {original_bits}/{compressed_bits} = {ratio:.2f}:1")

    print(f"Size {direction} by {abs(change):.2f}% {emoji}!")

    codebook_file_size = os.path.getsize(codebookfilename)
    compressed_file_size = os.path.getsize(comprfilename)

    print("\n಄ Actual generated file sizes:")
    print(f"   ❥ Codebook JSON: {codebook_file_size} bytes")
    print(f"   ❥ Compressed BIN: {compressed_file_size} bytes")

    print("\n─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")

    return ratio

def read_image(name, mode):
    img = Image.open(name).convert(mode)
    img_array = np.array(img)
    return img_array

def pad_image(img_array, block_height, block_width):
    height, width = img_array.shape[:2]
    padding_width = (block_width - (width % block_width)) % block_width
    padding_height = (block_height - (height % block_height)) % block_height

    padWidth = (0, padding_height), (0, padding_width)
    if img_array.ndim == 3:
        padWidth += ((0, 0),)
    
    padded_img = np.pad(img_array, pad_width = padWidth, mode='edge')
    
    return padded_img, (height, width)

def extract_vectors(image, block_height, block_width):
    height, width = image.shape[:2]
    vectors = []
    x = 0
    y = 0
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
    codebook = np.mean(vectors, axis = 0, keepdims = True)
    while len(codebook) < K:
        codebook = split_codebook(codebook)

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

    codebook = np.asarray(codebook, dtype=float)

    K = len(codebook)

    codebook = np.clip(np.rint(codebook), 0, 255).astype(np.uint8)

    if mode == "L":
        codebook = codebook.reshape(K, vector_height, vector_width)

    elif mode == "RGB":
        codebook = codebook.reshape(K, vector_height, vector_width, 3)

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

def compress(img, blockHeight, blockWidth, K, codebookfilename, comprfilename, mode):
    img = read_image(img, mode)
    orgHeight, orgWidth = img.shape[:2]
    paddedimg, _ = pad_image(img , blockHeight, blockWidth)
    vectors = extract_vectors(paddedimg, blockHeight, blockWidth)
    codebook = generate_codebook(vectors, K)

    codebook = np.clip(np.rint(codebook), 0, 255).astype(np.uint8)
    write_codebook(codebookfilename, codebook, orgHeight, orgWidth, blockHeight, blockWidth, mode)
    
    _, indices = assign_vectors(vectors, codebook)
    with open(comprfilename, "wb") as file:
        write_vector_bitstream(file, indices, K)


def run_encoding():

    print("\n─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")
    print("⋆⋅☆⋅⋆       Vector Quantization Compression       ⋆⋅☆⋅⋆")
    print("─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")

    try:

        image_path = input("\n⚝ Enter image filename: ").strip()

        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image file not found.")

        print("\n⚝ Select image mode:")
        print("   1- Grayscale")
        print("   2- RGB")

        mode_choice = input("Enter choice: ").strip()
        if mode_choice == "1":
            mode = "L"
        elif mode_choice == "2":
            mode = "RGB"

        else:
            raise ValueError("Invalid image mode selection.")
    
        block_height = int(input("⚝ Enter block height: "))
        block_width = int(input("⚝ Enter block width: "))

        if block_height <= 0 or block_width <= 0:
            raise ValueError("Block dimensions must be positive.")

        K = int(input("⚝ Enter number of codebook vectors (K): "))

        if K < 1:
            raise ValueError("K must be larger than 1")

        img = read_image(image_path, mode)

        height, width = img.shape[:2]

        padded_height = ((height + block_height - 1) // block_height) * block_height

        padded_width = ((width + block_width - 1) // block_width) * block_width

        number_of_blocks = ((padded_height // block_height) * (padded_width // block_width))

        if K > number_of_blocks:
            raise ValueError("K cannot exceed the number of image blocks.")

        codebook_filename = input("⚝ Enter what you'd like to name codebook file (include .json): ")
        compressed_filename = input("⚝ Enter what you'd like to name compressed image file (include .bin): ")

        compress(image_path, block_height, block_width, K, codebook_filename, compressed_filename, mode)
        getRatio(codebook_filename, compressed_filename)

        if not os.path.isfile(codebook_filename):
            raise RuntimeError("Codebook file was not created.")

        if not os.path.isfile(compressed_filename):
            raise RuntimeError("Compressed image file was not created.")

        print("\n಄ Compression completed successfully!")

        print("\n಄ Image Information:")
        print("   • Original dimensions:", height, "x", width)
        print("   • Padded dimensions:", padded_height, "x", padded_width)
        print("   • Block dimensions:", block_height, "x", block_width)
        print("   • Number of blocks:", number_of_blocks)
        print("   • Codebook size:", K)

        print("\n಄ Generated Files:")
        print(
            "   ❥", codebook_filename,
            "-", os.path.getsize(codebook_filename), "bytes"
        )
        print(
            "   ❥", compressed_filename,
            "-", os.path.getsize(compressed_filename), "bytes"
        )

        print("\n─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───")

    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:

        print("\n✮⋆˙ Compression Error:", error)
        print("Please try again.")

if __name__ == "__main__":
    run_encoding()