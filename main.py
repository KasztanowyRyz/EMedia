import sys
import struct
import subprocess
from multiprocessing import Process, Pipe
import chardet
import numpy as np
import cv2
import matplotlib
import zlib
import math
from chunk_model import ChunkModel, critical_chunks as c_c_dict
import os
import _rsa

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from png_funs import read_chunks


def encrypt_image(image_name, IHDR, IDAT, IEND, public_key):
    file_name = image_name[:-4] + "_enc.png"

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        IDAT_length = 0
        IDAT_data = b""
        block_size = _rsa.BLOCK_SIZE
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]

            cipherbytes = _rsa.rsa_encrypt(block, public_key)
            IDAT_length += len(cipherbytes)
            IDAT_data += cipherbytes

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if IHDR != None:
            IHDR.write_to_file(file)

        if newIDAT != None:
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)
    return newIDAT


def decrypt_image(image_name, IHDR, IDAT, IEND, private_key):
    file_name = image_name[:-4] + "_dec.png"

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        if IHDR != None:
            IHDR.write_to_file(file)

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        block_size = _rsa.BITS // 8

        IDAT_length = 0
        IDAT_data = b""
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]
            # Process the block as desired
            # For example, print the block contents
            bytes_text = _rsa.rsa_decrypt(block, private_key)
            IDAT_length += len(bytes_text)
            IDAT_data += bytes_text

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if newIDAT != None:
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)


def decompress(IDAT):
    IDAT_dc_data = zlib.decompress(IDAT.data)
    IDAT_length = len(IDAT_dc_data)
    IDAT_crc = zlib.crc32(b"IDAT" + IDAT_dc_data).to_bytes(4, byteorder="big")

    newIDAT = ChunkModel(
        IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_dc_data, IDAT_crc
    )

    return newIDAT


def compress(IDAT_dc):
    IDAT_data = zlib.compress(IDAT_dc.data)
    IDAT_length = len(IDAT_data)
    IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

    newIDAT = ChunkModel(
        IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
    )

    return newIDAT


def encrypt_decompressed(image_name, IHDR, IDAT, IEND, public_key):
    file_name = image_name[:-4] + "_enc_dc.png"

    IDAT = decompress(IDAT)

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        IDAT_length = 0
        IDAT_data = b""
        block_size = _rsa.BLOCK_SIZE
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]

            cipherbytes = _rsa.rsa_encrypt(block, public_key)
            IDAT_length += len(cipherbytes)
            IDAT_data += cipherbytes

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if IHDR != None:
            IHDR.write_to_file(file)

        if newIDAT != None:
            newIDAT = compress(newIDAT)
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)
    return newIDAT


def decrypt_decompressed(image_name, IHDR, IDAT, IEND, private_key):
    file_name = image_name[:-4] + "_dec_dc.png"

    IDAT = decompress(IDAT)

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        if IHDR != None:
            IHDR.write_to_file(file)

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        block_size = _rsa.BITS // 8

        IDAT_length = 0
        IDAT_data = b""
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]
            # Process the block as desired
            # For example, print the block contents
            bytes_text = _rsa.rsa_decrypt(block, private_key)
            IDAT_length += len(bytes_text)
            IDAT_data += bytes_text

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if newIDAT != None:
            newIDAT = compress(newIDAT)
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)


def divide_image(chunks):
    IHDR = None
    PLTE = None
    IDAT = None
    IEND = None
    IDAT_length = 0
    IDAT_data = b""
    for chunk in chunks:
        if chunk.type_bytes == b"IHDR":
            IHDR = ChunkModel(
                chunk.length_bytes, chunk.type_bytes, chunk.data, chunk.crc
            )
        if chunk.type_bytes == b"PLTE":
            PLTE = ChunkModel(
                chunk.length_bytes, chunk.type_bytes, chunk.data, chunk.crc
            )
        if chunk.type_bytes == b"IEND":
            IEND = ChunkModel(
                chunk.length_bytes, chunk.type_bytes, chunk.data, chunk.crc
            )
        if chunk.type_bytes == b"IDAT":
            IDAT_length += int.from_bytes(chunk.length_bytes, byteorder="big")
            IDAT_data += chunk.data

    IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

    IDAT = ChunkModel(
        IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
    )

    return IHDR, PLTE, IDAT, IEND


def img_info(IHDR, IDAT):
    width, height, bitd, colort, compm, filterm, interlacem = struct.unpack(
        ">IIBBBBB", IHDR.data
    )
    if colort == 2:
        bytes_per_pixel = 3
    elif colort == 6:
        bytes_per_pixel = 4
    else:
        raise ValueError("Not an RGB(A) png file")

    expected_len = height * (1 + width * bytes_per_pixel)
    print(
        f"Width: {width} Height: {height} Color type: {colort} Expected len: {expected_len}"
    )

    print("Compressed len: " + str(int.from_bytes(IDAT.length_bytes, byteorder="big")))

    IDAT_decompressed = zlib.decompress(IDAT.data)

    print("Decompressed len: " + str(len(IDAT_decompressed)))

    return IDAT_decompressed


def encrypt_image_ctr(image_name, IHDR, IDAT, IEND, public_key):
    file_name = image_name[:-4] + "_enc_ctr.png"

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        IDAT_length = 0
        IDAT_data = b""
        block_size = _rsa.BLOCK_SIZE
        counter = 0
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]

            cipherbytes = _rsa.rsa_encrypt_ctr(block, public_key, counter)
            IDAT_length += len(cipherbytes)
            IDAT_data += cipherbytes
            counter += 1

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if IHDR != None:
            IHDR.write_to_file(file)

        if newIDAT != None:
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)
    return newIDAT


def decrypt_image_ctr(image_name, IHDR, IDAT, IEND, private_key):
    file_name = image_name[:-4] + "_dec_ctr.png"

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        if IHDR != None:
            IHDR.write_to_file(file)

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        block_size = _rsa.BITS // 8

        IDAT_length = 0
        IDAT_data = b""
        counter = 0
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]
            # Process the block as desired
            # For example, print the block contents
            bytes_text = _rsa.rsa_encrypt_ctr(block, private_key, counter)
            IDAT_length += len(bytes_text)
            IDAT_data += bytes_text
            counter += 1

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if newIDAT != None:
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)


def encrypt_decompressed_ctr(image_name, IHDR, IDAT, IEND, public_key):
    file_name = image_name[:-4] + "_enc_dc_ctr.png"

    IDAT = decompress(IDAT)

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        IDAT_length = 0
        IDAT_data = b""
        block_size = _rsa.BLOCK_SIZE
        counter = 0
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]

            cipherbytes = _rsa.rsa_encrypt_ctr(block, public_key, counter)
            IDAT_length += len(cipherbytes)
            IDAT_data += cipherbytes
            counter += 1

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if IHDR != None:
            IHDR.write_to_file(file)

        if newIDAT != None:
            newIDAT = compress(newIDAT)
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)
    return newIDAT


def decrypt_decompressed_ctr(image_name, IHDR, IDAT, IEND, public_key):
    file_name = image_name[:-4] + "_dec_dc_ctr.png"

    IDAT = decompress(IDAT)

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")

        if IHDR != None:
            IHDR.write_to_file(file)

        data_length = int.from_bytes(IDAT.length_bytes, byteorder="big")
        block_size = _rsa.BITS // 8

        IDAT_length = 0
        IDAT_data = b""
        counter = 0
        for i in range(0, data_length, block_size):
            block = IDAT.data[i : i + block_size]

            cipherbytes = _rsa.rsa_encrypt_ctr(block, public_key, counter)
            IDAT_length += len(cipherbytes)
            IDAT_data += cipherbytes
            counter += 1

        IDAT_crc = zlib.crc32(b"IDAT" + IDAT_data).to_bytes(4, byteorder="big")

        newIDAT = ChunkModel(
            IDAT_length.to_bytes(4, byteorder="big"), b"IDAT", IDAT_data, IDAT_crc
        )

        if newIDAT != None:
            newIDAT = compress(newIDAT)
            newIDAT.write_to_file(file)

        if IEND != None:
            IEND.write_to_file(file)


def main():
    choice = None
    if len(sys.argv) != 1:
        img = "./RSA/" + sys.argv[1]
    else:
        img = None

    _, critical_chunks = read_chunks(img)
    IHDR, PLTE, IDAT, IEND = divide_image(critical_chunks)
    img_info(IHDR, IDAT)

    private_key, public_key = _rsa.generate_rsa_key_pair()
    _rsa.write_keys_to_file(private_key, public_key, "./RSA/_keys.txt")
    private_key, public_key = _rsa.read_keys_from_file("./RSA/_keys.txt")

    newIDAT = encrypt_image(img, IHDR, IDAT, IEND, public_key)
    print("decrypt")
    decrypt_image(
        img,
        IHDR,
        newIDAT,
        IEND,
        private_key,
    )

    newIDAT = encrypt_decompressed(img, IHDR, IDAT, IEND, public_key)
    print("decrypt")
    decrypt_decompressed(
        img,
        IHDR,
        newIDAT,
        IEND,
        private_key,
    )

    # Generate key pair
    # private_key, public_key = rsa.generate_rsa_key_pair()

    # encrypt_image(img, public_key)
    # decrypt_image(f"{img}", private_key)
    # os.system('cls' if os.name == 'nt' else 'clear')
    # while(True):
    #     print("Menu -- Choose option: \n")
    #     print("1. Read metadata \n")
    #     print("2. Show image pack (image, FFT, etc.) \n")
    #     print("3. Anonymize image \n")
    #     print("4. Choose image \n")
    #     print("0. Exit \n")
    #     choose = input("Your choice: ")
    #     if (choose == '1' and img is not None):
    #         read_png(img)
    #     elif (choose == '2' and img is not None):
    #         show_png(img)
    #     elif (choose == '3' and img is not None):
    #         anonymize_image(img, True)
    #     elif (choose == '4'):
    #         img = input("Provide image name/path: ")
    #     elif (choose == '0'):
    #         exit()
    #     else:
    #         print("Not recognized option or no image found\n")


if __name__ == "__main__":
    main()
