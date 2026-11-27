from Decoder import run_decoding


def main():

    print("===== Vector Quantization =====")
    print("1. Encoding")
    print("2. Decoding")

    choice = input("Choose an operation: ")

    if choice == "1":
        print("Encoding will be implemented here.")

        # TODO: Call the encoding function here

    elif choice == "2":
        run_decoding()

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()