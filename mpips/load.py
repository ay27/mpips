#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os


def load(input_file, rank, num):
    # if os.path.isdir(input_file):
    #     return load_dir(input_file, rank, num)
    # else:
    return load_single_file(input_file, rank, num)


def get_read_bytes_single(tot_bytes, rank, num):
    each_base, rem = tot_bytes / num, tot_bytes % num
    if rank < rem:
        return (each_base + 1) * rank, each_base + 1
    else:
        return (each_base + 1) * rem + (rank - rem) * each_base, each_base


def load_single_file(input_file, rank, num):
    with open(input_file) as f:
        f.seek(0, 2)
        tot_bytes = f.tell()
        f.seek(0, 0)
        start_offset, read_bytes = get_read_bytes_single(tot_bytes, rank, num)
        f.seek(start_offset, 0)
        raw_data = f.read(int(read_bytes))
        if rank != 0:
            # push back head data
            first_new_line_pos = raw_data.find("\n")
            raw_data = raw_data[first_new_line_pos + 1:]
        if rank != (num - 1):
            # read more tail data
            buf = f.readline()
            raw_data = "".join([raw_data, buf])
    return raw_data.strip().split("\n")
