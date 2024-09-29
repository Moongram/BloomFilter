from sklearn.utils import murmurhash3_32
from random import randint, sample
import math
from bitarray import bitarray
import random
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np

# Type ./run.sh in the current directory to run the script
# Set fixed random seeds
RANDOM_SEED = 3
random.seed(RANDOM_SEED)

def hashfunc(m):
    random_seed = randint(0, 2**32 - 1)
    def hash(key):
        hash_val = murmurhash3_32(key, seed=random_seed)
        return hash_val % m
    return hash

class BloomFilter():
    def __init__(self, n, fp_rate):
        # table size
        self.n = n
        # false positive rate
        self.fp_rate = fp_rate
        r = abs((self.n * math.log(self.fp_rate)) / math.log(0.618))
        # size of the bit array
        self.r = 1 << (math.ceil(r) - 1).bit_length()
        self.bit_array = bitarray(self.r)
        self.bit_array.setall(0)
        # number of hash functions
        self.k = math.ceil((self.r / self.n) * math.log(2))
        # hash functions
        self.h = [hashfunc(self.r) for _ in range(self.k)]


    def insert(self, key):
        for i in range(self.k):
            self.bit_array[self.h[i](key)] = 1

    def test(self, key):
        return all(self.bit_array[self.h[i](key)] for i in range(self.k))
    
    def update_extended_filter(self):
        self.k = math.ceil(0.7 * (self.r / self.n))
        self.h = [hashfunc(self.r) for _ in range(self.k)]

    def getBitSize(self):
        return self.r
    
def generate_datasets():
    membership_set = set(sample(range(10000, 100000), 10000))
    # Manipulation to get all non_members
    non_members_list = list(set(range(10000, 100000)) - membership_set)

    test_set_non_members_list = sample(non_members_list, 1000)
    test_set_members_list = sample(list(membership_set), 1000)

    test_set_non_members_list.extend(test_set_members_list)
    return list(membership_set), test_set_non_members_list

def warmup_test_res():
    membership_set, test_set = generate_datasets()
    return calculate_fp_rate([0.01, 0.001, 0.0001], membership_set, test_set, False)[0]

def calculate_fp_rate(rates, membership_set, test_set, is_extended):
    res = []
    bf_list = []
    mem_size = []
    theo_bf_size = []
    for rate in rates:
        if is_extended:
            bf = BloomFilter(377871, rate)
            bf.update_extended_filter()
            theo_bf_size.append(bf.getBitSize() // 8)
        else:
            bf = BloomFilter(len(membership_set), rate)
        for item in membership_set:
            bf.insert(item)
        bf_list.append(bf)    
        fp = 0
        if (not is_extended):
            for item in test_set[:1000]:  # First 1000 are non-members
                if bf.test(item):
                    fp += 1
            actual_fp_rate = fp / 1000
        else:
            for item in test_set:
                if bf.test(item):
                    fp += 1
            actual_fp_rate = fp / 2000
            mem_size.append(sys.getsizeof(bf.bit_array))
        res.append(actual_fp_rate)
    return res, mem_size, theo_bf_size

def export_warmup_results(real_fp_rates, filename):
    theoretical_fp_rates = [0.01, 0.001, 0.0001]
    
    with open(filename, 'w') as file:
        file.write("Theoretical FP        Real FP\n")
        
        for theo, real in zip(theoretical_fp_rates, real_fp_rates):
            file.write(f"{theo:<21} {real}\n")

def extended_practices():
    data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
    urllist = data.ClickURL.dropna().unique()
    test_url_list = sample(list(urllist), 1000)
    random_string_list = [str(randint(0, 1000000)) for _ in range(1000)]
    test_url_list.extend(random_string_list)
     # We want to vary R, which means vary fp rate, because R is calculated based on n and fp rate
    rates = [0.8, 0.7, 0.6, 0.5, 0.25, 0.05, 0.03, 0.01, 0.001, 0.0001]
    res = calculate_fp_rate(rates, urllist, test_url_list, True)
    py_ht = set(urllist)
    py_ht_mem_size = sys.getsizeof(py_ht)
    print("Memory usage from python hashtable to store urllist membership set: ", py_ht_mem_size)
    return res

def export_extended_results(res, filenamepl1):
    fp_rates = res[0]
    memory_usage = res[1]
    theoretical_bf_size = res[2]
    plt.figure(figsize=(10, 6))
    plt.plot(memory_usage, fp_rates, marker='o')
    plt.xlabel('Memory Usage (bytes)')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate vs Memory Usage for BF')
    plt.savefig(filenamepl1, format='png')
    plt.close()
    print("All of false positive rates we calculated:\n", fp_rates)
    print("Relative to above each fp rate, the real memory usage:\n", memory_usage)
    print("Relative to above each fp rate, theoretical memory usage:\n", theoretical_bf_size)

if __name__ == "__main__":
    # warmup main script, export to Results.txt
    export_warmup_results(warmup_test_res(), "Results.txt")
    # extended practices, export plot to fp_rate_vs_memory.png
    export_extended_results(extended_practices(), "fp_rate_vs_memory.png")