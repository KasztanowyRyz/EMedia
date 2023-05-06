# CC - critical chunk | AC - ancillary chunk
chunk_types = {
    "49484452": "IHDR",  # CC
    "504c5445": "PLTE",  # CC
    "49444154": "IDAT",  # CC
    "49454e44": "IEND",  # CC
    "73524742": "sRGB",  # AC
    "67414D41": "gAMA",  # AC
    "70485973": "pHYs",  # AC
    "73424954": "sBIT",  # AC
    "73504C54": "sPLT",  # AC
    "74494D45": "tIME",  # AC
    "6348524D": "cHRM",  # AC
    "74455874": "tEXt",  # AC
    "69545874": "iTXt",  # AC
}

critical_chunks = {
    "49484452": "IHDR",  # CC
    "504c5445": "PLTE",  # CC
    "49444154": "IDAT",  # CC
    "49454e44": "IEND",  # CC
}


class ChunkModel:
    length_bytes = b""
    type_bytes = b""
    data = b""
    crc = b""

    def __init__(self, length_bytes, type_bytes, data, crc):
        self.length_bytes = length_bytes
        self.type_bytes = type_bytes
        self.data = data
        self.crc = crc

    def __str__(self):
        length = int.from_bytes(self.length_bytes, byteorder="big")
        type = chunk_types.get(self.type_bytes.hex(), "unknown")
        crc = self.crc.hex()

        if type == "unknown":
            return f"Type:{type} Length:{length}"

        return f"Type:{type}    Length:{length}"

    def writeToFile(self, file):
        file.write(self.length_bytes)
        file.write(self.type_bytes)
        file.write(self.data)
        file.write(self.crc)
