import sys, os
sys.path.append(os.path.dirname(__file__))
import math

def conv_output(size, kernel, stride, padding):
    sliding_size = size + 2*padding - kernel
    return math.floor(sliding_size/stride + 1)
