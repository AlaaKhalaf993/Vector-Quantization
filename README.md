# Vector-Quantization

Implementation of Vector Quantization image compression for grayscale and
RGB images.

## Encode an image

Run the encoder directly:

```text
python Encoding.py
```

Or run `python Main.py` and choose `1`. Enter the image filename, vector
(block) height, vector width, and codebook size `K` when prompted. The encoder
supports Pillow `L` (grayscale) and `RGB` images. `K` must be a positive
integer; it does not need to be a power of two.

If the image dimensions are not divisible by the vector dimensions, the
encoder pads the bottom and right edges. The decoder removes this padding
when reconstructing the original dimensions.

Encoding creates `codebook.json` and `compressed_image.bin` in the current
working directory. The JSON stores the image mode, original dimensions,
vector dimensions, codebook size, and codebook vectors. Grayscale pixels are
integers; RGB pixels are `[R, G, B]` lists.

The binary file contains only codebook indices in row-major order. Each index
uses `(K - 1).bit_length()` bits, packed MSB-first with zero padding at the
end of the final byte when needed. For `K = 1`, the binary file is empty.

## Decode an image

Run the decoder directly:

```text
python Decoding.py
```

Or run `python Main.py` and choose `2`. Enter the codebook JSON filename and
compressed-index binary filename when prompted (for example,
`codebook.json` and `compressed_image.bin`).

The decoder saves `output/reconstructed_image.png` and
`output/reconstructed_image.json`.

## Example

Run `python Encoding.py`, then enter these values at the prompts for a
grayscale image:

```text
sample_gray.png
2
2
3
```

For an RGB image, run the encoder again and enter:

```text
sample_rgb.png
2
2
4
```

After encoding, run `python Decoding.py` and provide the generated
`codebook.json` and `compressed_image.bin` files.
