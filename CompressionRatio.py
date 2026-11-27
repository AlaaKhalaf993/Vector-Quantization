import os


def calculate_compression_ratio(original_file, *compressed_files):
    if not compressed_files:
        raise ValueError("at least one compressed file is required")

    original_size = os.path.getsize(original_file)
    compressed_size = sum(os.path.getsize(file) for file in compressed_files)

    if original_size == 0:
        raise ValueError("original file must not be empty")
    if compressed_size == 0:
        raise ValueError("compressed files must not be empty")

    return original_size / compressed_size
