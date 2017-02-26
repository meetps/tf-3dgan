import numpy as np


def testOFFReader():
    path = '../sample-data/chair.off'
    raw_data = tuple(open(path, 'r'))
    header = raw_data.strip(' ')[:-1]
    n_vertices, n_faces = header[0], header[1]
        

if __name__ == '__main__':
    a = testOFFReader()
    print a
