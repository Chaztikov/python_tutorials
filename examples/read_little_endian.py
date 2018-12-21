'''
https://stackoverflow.com/questions/46234200/python-read-hex-little-endian-do-math-with-it-and-overwrite-the-result-to-the

you can use the struct library to handle little endian as shown in the doc at https://docs.python.org/3/library/struct.html#struct.pack_into

For your specific task, I don't know if I understood correctly, because you didn't specify what kind of data do you have in your binary file... let's assume we have signed 32 bit integers, my code would be something like this:
'''
import struct

# we are assuming you have 32 bit integers on your file
block_size = 4
filename = "prova.bin"


# function to do "some math... :)"
def do_some_math(my_hex_value):
    return my_hex_value + 1


# open and read the whole file
with open(filename, "r+b") as f:
    my_byte = f.read(block_size)
    while(len(my_byte) == block_size):

        # unpack the 4 bytes value read from file
        # "<" stands for "little endian"
        # "i" stands for "integer"
        # more info on the struct library in the official doc
        my_hex_value = struct.unpack_from("<i", my_byte)[0]

        print("Before math = " + str(my_hex_value))

        # let's do some math
        my_hex_value = do_some_math(my_hex_value)

        print("After math = " + str(my_hex_value))

        # let's repack the hex back
        my_byte = struct.pack("<i", my_hex_value)

        # let's reposition the file pointer so as to overwrite
        # the bytes we have previously read 
        f.seek(f.tell() - block_size)

        # let's override the old bytes
        f.write(my_byte)

        # let's read another chunk to repeat till the eof
        my_byte = f.read(block_size)
