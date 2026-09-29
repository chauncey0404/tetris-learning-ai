from tetrio.control.input_controller import windows_input_abi

def main():
    info = windows_input_abi()
    print("Windows SendInput ABI")
    print("=" * 48)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    if info["platform"] == "nt":
        expected = 40 if info["is_64_bit"] else 28
        print(f"expected INPUT size : {expected}")
        print("ABI status           : " + (
            "PASS" if info["input_size"] == expected else "FAIL"
        ))

if __name__ == "__main__":
    main()
