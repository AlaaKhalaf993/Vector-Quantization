from Decoding import run_decoding
from Encoding import run_encoding

def main():

    print("─── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ──")
    print("⋆⋅☆⋅⋆Welcome to our Vector Quantization compression/decompression simulation!⋆⋅☆⋅⋆")
    while True:
        print("\n಄ 1- Compress")
        print("಄ 2- Decompress")
        print("಄ 3- Exit")

        try:
            choice = int(input("⚝ Enter the number corresponding to the command you'd like: "))
        except ValueError:
            print("✮⋆˙ Please enter a valid number!")
            continue
        
        match choice:
            case 1:
                run_encoding()

            case 2:
                run_decoding()

            case 3:
                print("❥ Goodbye!")
                break

            case _:
                print("✮⋆˙ Error: You seem to have typed a non-viable command, please try again!")


if __name__ == "__main__":
    main()