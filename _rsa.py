import random
from math import gcd
import libnum
import rsa

BITS = 32
BLOCK_SIZE = 2
NONCE = 123456


def write_keys_to_file(private_key, public_key, filename):
    with open(filename, "w") as file:
        file.write(str(private_key) + "\n")
        file.write(str(public_key) + "\n")
    print("Keys successfully written to", filename)


def read_keys_from_file(filename):
    with open(filename, "r") as file:
        private_key = file.readline().rstrip("\n")
        public_key = file.readline().rstrip("\n")
    return eval(private_key), eval(public_key)


def generate_rsa_key_pair():
    n = 0
    while n.bit_length() != BITS:
        p = libnum.generate_prime(BITS // 2 + 1)
        q = libnum.generate_prime(BITS // 2 - 1)

        n = p * q
    print(p)
    print(q)
    print(f"Key length: {n.bit_length()}")

    phi_n = (p - 1) * (q - 1)

    e = choose_public_exponent(phi_n)

    d = modular_inverse(e, phi_n)

    public_key = (e, n)
    private_key = (d, n)
    return private_key, public_key


def choose_public_exponent(phi_n):
    while True:
        e = random.randint(2, phi_n)
        if gcd(e, phi_n) == 1:
            return e


def modular_inverse(a, m):
    t1, t2 = 0, 1
    r1, r2 = m, a
    while r2 != 0:
        quotient = r1 // r2
        t1, t2 = t2, t1 - quotient * t2
        r1, r2 = r2, r1 - quotient * r2
    if r1 > 1:
        raise ValueError("a is not invertible")
    if t1 < 0:
        t1 += m
    return t1


def rsa_encrypt(plaintext_bytes, public_key):  # ECB
    e, n = public_key
    plaintext_int = int.from_bytes(plaintext_bytes, byteorder="big")
    ciphertext = pow(plaintext_int, e, n)
    return ciphertext.to_bytes(BITS // 8, "big")


def rsa_decrypt(ciphertext, private_key):  # ECB
    d, n = private_key
    ciphertext_int = int.from_bytes(ciphertext, byteorder="big")
    plaintext_bytes = pow(ciphertext_int, d, n)
    return plaintext_bytes.to_bytes(BLOCK_SIZE, "big")


def rsa_encrypt_ctr(plaintext_bytes, public_key, counter):  # ECB
    e, n = public_key
    plaintext_int = int.from_bytes(plaintext_bytes, byteorder="big")
    a = counter + NONCE
    ciphertext = pow(a, e, n)
    msg = ciphertext ^ plaintext_int
    return msg.to_bytes(BITS // 8, "big")


def rsa_encrypt_lib(plaintext_bytes, public_key):  # ECB
    e, n = public_key
    ciphertext = rsa.encrypt(plaintext_bytes, rsa.PublicKey(n, e))
    return ciphertext


def rsa_decrypt_lib(ciphertext, private_key, public_key):  # ECB
    d, n = private_key
    e, n = public_key
    plaintext_bytes = rsa.decrypt(
        ciphertext,
        rsa.PrivateKey(
            n,
            e,
            d,
            14384688583149222179207769174462437258609118735141613223494656870670526608633876800301335611075352762380197760438999012385407537969683215147318279996339443,
            6375649837373070282098372875364179864225074220340178255863735221109989893238185738380431671085177012053046007524178168595837338198461570581609789917181217,
        ),
    )
    return plaintext_bytes
