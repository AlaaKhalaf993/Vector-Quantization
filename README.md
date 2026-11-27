# Vector-Quantization
Implementation of the Vector Quantization compression technique.

## Decode an image

Run `python Main.py` and choose `2`, or run `python Decoder.py` directly. Enter
the codebook JSON filename and compressed-index binary filename when prompted.

The JSON file must contain `mode`, `original_height`, `original_width`,
`vector_height`, `vector_width`, `number_of_codebooks`, and `codebook`.
Codebook keys are consecutive indices starting at `"0"`. The binary file
contains only those indices, packed in row-major order, most-significant bit
first. Each index uses `(number_of_codebooks - 1).bit_length()` bits; any final
byte padding consists of zero bits. A one-entry codebook uses an empty binary
file.

The decoder saves `output/reconstructed_image.png` and
`output/reconstructed_image.json`.
