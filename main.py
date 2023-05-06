import sys
import struct
import subprocess
from multiprocessing import Process, Pipe
import chardet
import numpy as np
import matplotlib
import zlib
from chunk_model import ChunkModel, critical_chunks as c_c_dict

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# TODO: Clean up read_png function-> args(chunks) then print their data
# TODO: Group all similiar functions into different files


def spectrum_png(image):
    img = plt.imread(image)
    # Apply Fourier Transform on the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Plot the magnitude spectrum
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()

    # Apply Inverse Fourier Transform on the frequency domain representation
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Compare the original and reconstructed image
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Input Image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap="gray")
    plt.title("Reconstructed Image"), plt.xticks([]), plt.yticks([])
    plt.show()


def show_png(image):
    img = plt.imread(image)
    plt.imshow(img)
    plt.show()

    # output = subprocess.check_output(["display", image])
    # print(output.decode("utf-8"))


def print_palette(palette):
    print("Palette:")

    counter = 0
    for item in list(palette):
        print(item, end=" ")
        counter += 1
        if counter % 3 == 0:
            print("\n")
    print("\n")


def read_chunks(image, connection):
    # Open the PNG file in binary mode
    with open(image, "rb") as file:
        # Check if the file is a valid PNG file
        if file.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Not a valid PNG file")

        # Read the chunks of the PNG file
        chunks = []
        critical_chunks = []
        while True:
            length_bytes = file.read(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder="big")
            type_bytes = file.read(4)
            data = file.read(length)
            crc = file.read(4)

            chunk = ChunkModel(length_bytes, type_bytes, data, crc)

            chunks.append(chunk)

            type = c_c_dict.get(type_bytes.hex(), "AC")
            # Check if chunk is critical
            if type != "AC":
                critical_chunks.append(chunk)

        # Return readed chunks
        data = (chunks, critical_chunks)
        connection.send(data)


def anonymize_image(image_name, critical_chunks, join=False):
    file_name = image_name[:-4] + "_anon.png"

    with open(file_name, "wb") as file:
        file.write(b"\x89PNG\r\n\x1a\n")
        # Optionally join IDAT chunks
        if join:
            IHDR = None
            PLTE = None
            IDAT = None
            IEND = None
            IDAT_length = 0
            IDAT_data = b""

            for chunk in critical_chunks:
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

            if IHDR != None:
                IHDR.write_to_file(file)

            if PLTE != None:
                PLTE.write_to_file(file)

            if IDAT != None:
                IDAT.write_to_file(file)

            if IEND != None:
                IEND.write_to_file(file)
        else:
            for chunk in critical_chunks:
                file.write(chunk.length_bytes)
                file.write(chunk.type_bytes)
                file.write(chunk.data)
                file.write(chunk.crc)


def read_png(image, connection):
    # Open the PNG file in binary mode
    with open(image, "rb") as file:
        # Check if the file is a valid PNG file
        if file.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Not a valid PNG file")

        # Read the chunks of the PNG file
        chunks = []
        while True:
            length_bytes = file.read(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder="big")
            type_bytes = file.read(4)
            data = file.read(length)
            crc = file.read(4)
            chunks.append((type_bytes, data, crc))

        # Extract the metadata from the IHDR chunk
        ihdr_chunk = next((chunk for chunk in chunks if chunk[0] == b"IHDR"), None)
        if ihdr_chunk:
            (
                width,
                height,
                bit_depth,
                color_type,
                compression_method,
                filter_method,
                interlace_method,
            ) = struct.unpack(">IIBBBBB", ihdr_chunk[1])
        else:
            print("iHDR chunk not found or invalid size")

        # Extract the metadata from the sRGB chunk
        srgb_chunk = next((chunk for chunk in chunks if chunk[0] == b"sRGB"), None)
        if srgb_chunk:
            rendering_intent = srgb_chunk[1][0]
            print(
                "Rendering intent:",
                rendering_intent,
                {
                    0: "perceptual",
                    1: "relative colorimetric",
                    2: "saturation",
                    3: "absolute colorimetric",
                }.get(rendering_intent, "unknown"),
            )
        else:
            print("sRGB chunk not found or invalid size")

        # Extract the metadata from the gAMA chunk
        gama_chunk = next((chunk for chunk in chunks if chunk[0] == b"gAMA"), None)
        if gama_chunk:
            gamma_value = struct.unpack(">I", gama_chunk[1])[0] / 100000.0
            print("Gamma value:", gamma_value)
        else:
            print("gAma chunk not found or invalid size")

        # Extract the metadata from the pHYs chunk
        phys_chunk = next((chunk for chunk in chunks if chunk[0] == b"pHYs"), None)
        if phys_chunk and len(phys_chunk[1]) >= 9:
            pixels_per_unit_x, pixels_per_unit_y, unit_specifier = struct.unpack(
                ">IIB", phys_chunk[1][:9]
            )
            print("Pixels per unit, X axis:", pixels_per_unit_x)
            print("Pixels per unit, Y axis:", pixels_per_unit_y)
            if unit_specifier == 1:
                print("Unit specifier: meter")
            elif unit_specifier == 0:
                print("Unit specifier: unknown")
        else:
            print("pHYs chunk not found or invalid size")

        # Extract the metadata from the EXIF chunk (if it exists)
        # exif_chunk = next((chunk for chunk in chunks if chunk[0] == b'eXIf'), None)
        # if exif_chunk:
        #    exif_data = exif_chunk[1][6:]
        #    if exif_data[:6] != b'Exif\x00\x00':
        #        raise ValueError('Not a valid EXIF data block')
        #    exif_metadata = {}
        #    for i in range(8, len(exif_data), 12):
        #        tag = exif_data[i:i + 2]
        #        format_img = exif_data[i + 2:i + 4]
        #        count = int.from_bytes(exif_data[i + 4:i + 8], byteorder='big')
        #        offset = int.from_bytes(exif_data[i + 8:i + 12], byteorder='big')
        #        value = exif_data[offset:offset + count]
        #        if format_img == b'\x02\x00':
        #            value = value.decode()
        #        elif format_img == b'\x03\x00':
        #            value = int.from_bytes(value, byteorder='big')
        #        exif_metadata[tag] = value

        # Print the metadata
        print("Width:", width, "pixels")
        print("Height:", height, "pixels")
        print("Bit depth:", bit_depth, " bits per channel")
        print(
            "Color type:",
            color_type,
            {
                0: "grayscale",
                2: "RGB",
                3: "indexed color",
                4: "grayscale with alpha",
                6: "RGB with alpha",
            }.get(color_type, "unknown"),
        )
        print(
            "Compression method:",
            compression_method,
            {0: "DEFLATE", 1: "INFLATE"}.get(compression_method, "unknown"),
        )
        print(
            "Filter method:",
            filter_method,
            {0: "Adaptive", 1: "Unknown"}.get(filter_method, "unknown"),
        )
        print(
            "Interlace method:",
            interlace_method,
            {0: "None", 1: "Adam7"}.get(interlace_method, "unknown"),
        )
        # if exif_chunk:
        #    print('EXIF metadata:', exif_metadata)
        # else:
        #    print('EXIF chunk not found or invalid size')

        # Extract the metadata from the sBIT chunk
        sbit_chunk = next((chunk for chunk in chunks if chunk[0] == b"sBIT"), None)
        if sbit_chunk:
            if color_type in (0, 4):
                gray_bits = struct.unpack(">B", sbit_chunk[1])[0]
                print("Significant bits per channel (gray):", gray_bits)
            elif color_type in (2, 6):
                red_bits, green_bits, blue_bits = struct.unpack(">BBB", sbit_chunk[1])
                print(
                    "Significant bits per channel (RGB):",
                    red_bits,
                    green_bits,
                    blue_bits,
                )
            elif color_type == 3:
                red_bits, green_bits, blue_bits = struct.unpack(">B", sbit_chunk[1][:3])
                alpha_bits = struct.unpack(">B", sbit_chunk[1][3:])[0]
                print(
                    "Significant bits per channel (palette):",
                    red_bits,
                    green_bits,
                    blue_bits,
                    alpha_bits,
                )

        # Extract the metadata from the PLTE chunk
        plte_chunk = next((chunk for chunk in chunks if chunk[0] == b"PLTE"), None)
        if plte_chunk:
            palette = struct.iter_unpack(">BBB", plte_chunk[1])

            print_palette(palette)
        else:
            # sPLT chunk be used for this purpose if colour types 2 and 6 is set
            plte_chunk = next((chunk for chunk in chunks if chunk[0] == b"sPLT"), None)
            if plte_chunk:
                palette = struct.iter_unpack(">BBB", plte_chunk[1])
                print("Palette:", list(palette))
            else:
                print("PLTE chunk not found or invalid size")

        # Extract the metadata from the tIME chunk
        time_chunk = next((chunk for chunk in chunks if chunk[0] == b"tIME"), None)
        if time_chunk:
            year, month, day, hour, minute, second = struct.unpack(
                ">HBBBBB", time_chunk[1]
            )
            print(
                "Creation time:",
                f"{year}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}",
            )
        else:
            print("tIme chunk not found or invalid size")

        # Extract the metadata from the cHRM chunk
        chrm_chunk = next((chunk for chunk in chunks if chunk[0] == b"cHRM"), None)
        if chrm_chunk:
            (
                white_point_x,
                white_point_y,
                red_x,
                red_y,
                green_x,
                green_y,
                blue_x,
                blue_y,
            ) = struct.unpack(">IIIIIIII", chrm_chunk[1])
            print("White point (x, y):", white_point_x, white_point_y)
            print("Red point (x, y):", red_x, red_y)
            print("Green point (x, y):", green_x, green_y)
            print("Blue point (x, y):", blue_x, blue_y)
        else:
            print("cHrm chunk not found or invalid size")

        # Extract the metadata from the text chunks
        for text_chunk in (chunk for chunk in chunks if chunk[0] == b"tEXt"):
            keyword, text_data = text_chunk[1].split(b"\0", 1)
            try:
                if keyword == b"XML:com.adobe.xmp":
                    # The XMP data is typically encoded using UTF-8
                    print("XMP data:", text_data.decode("utf-8"))
                else:
                    # Attempt to decode the text data using the detected encoding
                    detected_encoding = chardet.detect(text_data)["encoding"]
                    print(
                        "Text, keyword:",
                        keyword.decode(),
                        "data:",
                        text_data.decode(detected_encoding),
                    )
            except UnicodeDecodeError:
                print("Unable to decode text chunk")

        # Extract the metadata from the iTXt chunk
        itext_chunks = [chunk for chunk in chunks if chunk[0] == b"iTXt"]
        for chunk in itext_chunks:
            (
                keyword,
                compression_flag,
                compression_method,
                language_tag,
                translated_keyword,
                text,
            ) = chunk[1].split(b"\x00", 5)
            print("International text keyword:", keyword.decode())
            print("Compression flag:", compression_flag)
            print("Compression method:", compression_method)
            print("Language tag:", language_tag.decode())
            print("Translated keyword:", translated_keyword.decode())
            print("Text:", text.decode())

        # Extract the metadata from the IDAT chunk
        idat_chunk = next((chunk for chunk in chunks if chunk[0] == b"IDAT"), None)
        if idat_chunk:
            idat_length = len(idat_chunk[1])
            dec_data = struct.iter_unpack(f"{idat_length}B", idat_chunk[1])
            print("Decompressed data length:", idat_length)

        # Extract the metadata from the IEND chunk
        iend_chunk = next((chunk for chunk in chunks if chunk[0] == b"IEND"), None)
        if iend_chunk:
            print("IEND chunk found")
        else:
            print("IEND chunk not found")

        connection.send(chunks)


def main():
    conn1, conn2 = Pipe()

    read_process = Process(
        target=read_chunks,
        args=(
            sys.argv[1],
            conn2,
        ),
    )
    read_process.start()

    chunks, critical_chunks = conn1.recv()

    for chunk in critical_chunks:
        print(chunk)

    anonymize_image(sys.argv[1], critical_chunks, True)
    # read_chunks(sys.argv[1], 1)
    # result = result_queue.get(block=True, timeout=5)

    # print(value)

    # image_process = Process(target=show_png, args=(sys.argv[1],))
    # image_process.start()
    # plot_process = Process(target=spectrum_png, args=(sys.argv[1],))
    # plot_process.start()


if __name__ == "__main__":
    main()
