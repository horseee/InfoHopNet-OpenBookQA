#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import codecs
import csv
import functools
import random
import sys
import time

import progressbar
import argparse
import config

def progessbar(fn):
    @functools.wraps(fn)
    def wrapped(*args):
        widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.Timer()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=config.line_number).start()
        for i in range(config.line_number):
            fn(*args)
            bar.update(i + 1)
        bar.finish()
    return wrapped

@progessbar
def extract(language, reader, writer):
    line = next(reader)
    r, s, e = [s.split('/') for s in line[1:4]]
    if s[2] == language and e[2] == language:
        writer.writerow([r[2], s[3], e[3]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract infomation")
    parser.add_argument('--language', default = 'en')
    args = parser.parse_args()

    data_f = codecs.open(os.path.join(config.root_path, config.conceptnet_file), 'r', 'utf-8')
    write_f = codecs.open(os.path.join(config.root_path, 'conceptnet_'+ args.language+'.csv'), 'w', 'utf-8')
    data_reader = csv.reader(data_f, delimiter = '\t')
    data_writer = csv.writer(write_f, delimiter = '\t')

    extract(args.language,data_reader, data_writer)

    data_f.close()
    write_f.close()
