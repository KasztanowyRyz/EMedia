import random
from math import gcd
from math import sqrt
import sympy
import libnum

BITS = 1024
BLOCK_SIZE = 64


def generate_rsa_key_pair():
    n = 0
    while n.bit_length() != BITS:
        p = libnum.generate_prime(BITS // 2 + 1)
        q = libnum.generate_prime(BITS // 2 - 1)

        n = p * q

    print(f"Key length: {n.bit_length()}")

    phi_n = (p - 1) * (q - 1)

    e = choose_public_exponent(phi_n)

    d = modular_inverse(e, phi_n)

    public_key = (e, n)
    private_key = (d, n)
    return private_key, public_key


def generate_large_prime_number():
    while True:
        num = random.randint(2 ** (BITS - 1), 2**BITS - 1)
        if sympy.isprime(num):
            return num


def is_prime(n, k=10):
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2

    for _ in range(k):
        a = random.randint(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


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
