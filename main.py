import time
from functools import lru_cache
import heapq
from collections import Counter, defaultdict

import tqdm
from line_profiler_pycharm import profile

UTF8 = 'utf-8'
CP1252 = 'cp1252'
LATIN1 = 'latin-1'
base = 256
global num_of_bytes_per_compressed_char
num_of_bytes_per_compressed_char = 1


# get a string and return its lempel ziv encoded string
@profile
def lz_encode(string):
    words = defaultdict(int, {v: k for k, v in enumerate(sorted(set(string)))})
    words_len = len(words)
    start = 0
    alephbet = b''
    for char in words.keys():
        alephbet += char.encode("utf-8")
    alephbet = alephbet.decode("utf-8")
    alephbet += 2 * chr(0)
    global num_of_bytes_per_compressed_char
    compressed = []
    power = 2 ** (num_of_bytes_per_compressed_char * 8)
    for end in tqdm.trange(len(string)):
        section = string[start: end + 1]
        if section not in words:
            num = words[string[start: end]]
            new_word_code = number_to_string(num)
            compressed.append(new_word_code)
            words[section] = words_len
            words_len += 1
            start = end
            if words_len >= power and words_len % power == 0:
                print(f"num_of_bytes_per_compressed_char was increased to {num_of_bytes_per_compressed_char + 1}")
                compressed = "".join(compressed)
                compressed = chr(0).join(compressed[i:i + num_of_bytes_per_compressed_char] for i in
                                         range(0, len(compressed), num_of_bytes_per_compressed_char))
                compressed = chr(0) + compressed
                compressed = list(compressed)
                num_of_bytes_per_compressed_char += 1
                number_to_string.cache_clear()
                power = 2 ** (num_of_bytes_per_compressed_char * 8)

    compressed = "".join(compressed)
    last_word_code = number_to_string(words[string[start:]])
    compressed += last_word_code
    alephbet = str(num_of_bytes_per_compressed_char) + chr(0) + chr(0) + alephbet
    return compressed, alephbet


# get a lempel ziv encoded string and write its original string to the file
def lz_decode(string, words):
    words_len = len(words)
    words_opposite = {value: key for key, value in words.items()}
    count = 0
    code_word = ''
    word = ''
    final = []

    for char in string:
        count += 1
        if count % 1000000 == 0:
            print("lz_decode_to_file: " + str(count) + " chars done")

        code_word += char

        if len(code_word) == num_of_bytes_per_compressed_char:
            num = string_to_num(code_word)
            try:
                original_word = words_opposite[num]
            except KeyError:
                original_word = word + word[0]  # Handle new words in LZW

            final.append(original_word)
            word += original_word[0]

            if count > num_of_bytes_per_compressed_char:
                words_opposite[words_len] = word
                words_len += 1
                word = original_word

            code_word = ''  # Reset code_word after processing

    # Handle any remaining characters (partial code_word at the end)
    if code_word:
        try:
            num = string_to_num(code_word)
            if num in words_opposite:
                final.append(words_opposite[num])
        except:
            pass  # If the remaining characters don't form a valid code, we can skip

    return "".join(final)


# get a string and return its huffman encoded string
def huffman_encode(string):
    # Count frequency of each character
    chars_count = Counter(string)

    # Initialize the dictionary for Huffman encoding
    words_dict = {char: '' for char in chars_count}

    # Create a priority queue (min-heap) for the tree construction
    heap = [[weight, [char, '']] for char, weight in chars_count.items()]
    heapq.heapify(heap)

    # Build Huffman tree using a min-heap
    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])

    # Update words_dict with the final bit sequences
    for pair in heap[0][1:]:
        words_dict[pair[0]] = pair[1]

    # Encode the string into a binary representation
    compressed_bits = []
    for char in string:
        compressed_bits.append(words_dict[char])

    # Join bits into a single string
    compressed_bits = ''.join(compressed_bits)

    # Convert bits into bytes
    compressed = []
    for i in range(0, len(compressed_bits), num_of_bytes_per_compressed_char * 8):
        byte_chunk = compressed_bits[i:i + num_of_bytes_per_compressed_char * 8]
        if len(byte_chunk) < num_of_bytes_per_compressed_char * 8:
            byte_chunk = byte_chunk.ljust(num_of_bytes_per_compressed_char * 8, '0')
        decimal = int(byte_chunk, 2)
        compressed.append(number_to_string(decimal))

    # Combine the compressed bits into the final result
    compressed = ''.join(compressed)

    # Serialize the words_dict for use in decoding
    separator = num_of_bytes_per_compressed_char * chr(0)
    words_dict_string = ','.join(words_dict[char] for char in sorted(words_dict.keys()))

    return words_dict_string + separator + compressed


# get a huffman encoded string and return its original string
def huffman_decode(string):
    separator = num_of_bytes_per_compressed_char * chr(0)
    words_dict_string = string.split(separator)[0]
    values = words_dict_string.split(",")
    words_opposite = {values[i]: chr(i) for i in range(base)}
    compressed = string.split(separator)[1]
    threshold = max([len(key) for key, value in words_opposite.items()])
    count = 0
    code_word = ''
    compressed_bits = ''
    uncompressed = []
    for char in compressed:
        count += 1
        if count % 1000000 == 0:
            print("huffman_decode: " + str(count) + " chars done")
        code_word += char
        if count % num_of_bytes_per_compressed_char == 0:  # 24 bits
            num = string_to_num(code_word)
            s_number = f'0:0{num_of_bytes_per_compressed_char * 8}b'
            binary = ("{" + s_number + "}").format(num)
            # binary = '{0:024b}'.format(num)
            compressed_bits += binary
            compressed_bits_index = 0
            while len(compressed_bits) >= threshold:
                compressed_bits_index += 1
                binary_word = compressed_bits[:compressed_bits_index]
                if binary_word in words_opposite.keys():
                    uncompressed.append(words_opposite[binary_word])
                    compressed_bits = compressed_bits[compressed_bits_index:]
                    compressed_bits_index = 0
            code_word = ''
    uncompressed = "".join(uncompressed)
    if count % num_of_bytes_per_compressed_char == 0:
        return uncompressed
    compressed_bits_index = 0
    for i in range(threshold):
        compressed_bits_index += 1
        binary_word = compressed_bits[:compressed_bits_index]
        if binary_word in words_opposite.keys():
            uncompressed += words_opposite[binary_word]
            compressed_bits = compressed_bits[compressed_bits_index:]
            compressed_bits_index = 0
    return uncompressed


# get a string representation of a number (of 3 bytes)
@lru_cache(maxsize=None)
@profile
def number_to_string(number):
    # Calculate the number of bits in the number.
    # num_bits = num_of_bytes_per_compressed_char * 8
    # Precompute the bit masks for each 8-bit chunk.
    masks = [(number >> (8 * i)) & 0xFF for i in range(num_of_bytes_per_compressed_char)]
    # Convert each 8-bit chunk to a character using a bytearray.
    return ''.join(chr(mask) for mask in reversed(masks))

# revert the string (3 characters) back to the number
def string_to_num(string):
    # return ord(string[0]) * base * base + ord(string[1]) * base + ord(string[2])
    return sum([ord(i) * base ** e for e, i in enumerate(reversed(string))])


import pydivsufsort


def bwt_encode_large(s):
    """Encodes the input string using the Burrows-Wheeler Transform with an efficient suffix array."""
    s = s + '\0'  # Append null terminator
    suffix_arr = pydivsufsort.divsufsort(s)  # Efficient suffix array construction
    return ''.join(s[i - 1] if i > 0 else s[-1] for i in suffix_arr), suffix_arr.index(0)


def bwt_decode_large(r, index):
    """Decodes the Burrows-Wheeler Transform using an efficient method."""
    n = len(r)

    # Create the first column of the sorted rotations
    first_column = sorted(r)

    # Count occurrences and build the next array
    count = {}
    for char in r:
        count[char] = count.get(char, 0) + 1

    total = {char: 0 for char in count}
    next_array = [0] * n
    for i, char in enumerate(first_column):
        total[char] = total.get(char, 0) + 1
        pos = total[char] - 1
        next_array[pos] = i

    # Reconstruct the original string
    decoded = [''] * n
    row = index
    for i in range(n - 1, -1, -1):
        decoded[i] = r[row]
        row = next_array[row]

    return ''.join(decoded).rstrip('\0')  # Remove null terminator


def main():
    input_name = 'dickens'
    global num_of_bytes_per_compressed_char
    now = time.time()
    # read the original file, compress it with lz and huffman, and write it to dickens_compressed.txt
    with open(f'{input_name}.txt', 'r', encoding=CP1252, newline='\n') as file:
        string = file.read()
        string = "".join([i for i in string if i.isascii()])
        string = string.replace("", "")
        idx, bw_compressed = pydivsufsort.bw_transform(string)
        lz_compressed, alphabet = lz_encode("".join([chr(i) for i in bw_compressed]))
        huffman_len = len(lz_compressed)
        lz_and_huffman_compressed = huffman_encode(lz_compressed)
        lz_and_huffman_compressed = alphabet.encode("utf-8").decode("latin-1") + str(
            idx) + 2 * chr(0) + str(huffman_len) + 2 * chr(0) + lz_and_huffman_compressed
    with open(f'{input_name}_compressed_{num_of_bytes_per_compressed_char}.txt', "w",
              encoding=LATIN1, newline='\n') as file:
        file.write(lz_and_huffman_compressed)
    print("lz_and_huffman_compressed: " + str(time.time() - now) + " seconds")
    now = time.time()
    # read the compressed file from dickens_compressed.txt, decompress it, and write it to dickens_decompressed.txt
    with open(f'{input_name}_compressed_{num_of_bytes_per_compressed_char}.txt', 'r',
              encoding=LATIN1, newline='\n') as file:
        compressed = file.read()
    ab = ""
    for i, c in enumerate(compressed):
        if c == chr(0) and compressed[i + 1] == chr(0):
            break
        ab += c
    num_of_bytes_per_compressed_char = int(ab)
    compressed = compressed[i + 2:]

    alphabet = ""
    for i, c in enumerate(compressed):
        if c == chr(0) and compressed[i + 1] == chr(0):
            break
        alphabet += c
    alphabet = alphabet.encode("latin-1").decode("utf-8")
    words = {v: k for k, v in enumerate(alphabet)}

    compressed = compressed[i + 2:]

    idx = ""
    for i, c in enumerate(compressed):
        if c == chr(0) and compressed[i + 1] == chr(0):
            break
        idx += c
    idx = int(idx)
    compressed = compressed[i + 2:]

    huffman_len = ""
    for i, c in enumerate(compressed):
        if c == chr(0) and compressed[i + 1] == chr(0):
            break
        huffman_len += c
    huffman_len = int(huffman_len)
    compressed = compressed[i + 2:]

    huffman_decompressed = huffman_decode(compressed)
    lz_d = lz_decode(huffman_decompressed[:huffman_len], words)
    bw_decompressed = pydivsufsort.inverse_bw_transform(idx, bytes(lz_d, "ASCII"))
    with open(f'{input_name}_decompressed_{num_of_bytes_per_compressed_char}.txt', "w",
              encoding=UTF8, newline='\n') as file:
        file.write("".join([chr(i) for i in bw_decompressed]))
    print("lz_and_huffman_decompressed: " + str(time.time() - now) + " seconds")


if __name__ == '__main__':
    main()
