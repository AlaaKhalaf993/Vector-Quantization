from Decoding import run_decoding
from Encoding import run_encoding

def main():
    while True:
        print("===== Vector Quantization =====")
        print("1. Encoding")
        print("2. Decoding")
        print("3. Exit")

        choice = input("Choose an operation: ")

        if choice == "1":
            run_encoding()

        elif choice == "2":
            run_decoding()

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()