import torch
import hashlib

def vector_hash_color(vector):
    # Convert the PyTorch vector to a string
    vector_str = str(vector.detach().numpy())

    # Generate a hash of the string using SHA256
    hash_obj = hashlib.sha256(vector_str.encode())
    hash_str = hash_obj.hexdigest()

    # Convert the hash to RGB values
    r = int(hash_str[0:2], 16)
    g = int(hash_str[2:4], 16)
    b = int(hash_str[4:6], 16)

    # Normalize the RGB values to be between 0 and 1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Return the RGB values as a tuple
    return (r_norm, g_norm, b_norm)
