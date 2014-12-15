import numpy as np

class Point(object):

    def __init__(self, x, y):
        self.b = np.array([x, y])
        self.offsets = np.array([
            [+0, -1], [-1, -1], [-1, +0], [-1, +1],
            [+0, +1], [+1, +1], [+1, +0], [+1, -1]
        ])

    def __repr__(self):
        return str(self.b)

    def neighbors(self, c):
        offset = c - self.b
        index = 0
        for i in xrange(8):
            if np.array_equal(offset, self.offsets[i]):
                index = i
                break
        return iter(self.b + np.roll(self.offsets, -index, axis=0))

    def neighbor(self, idx):
        return self.b + self.offsets[idx]


def boundary_following(img):

    M, N = img.shape

    # Find starting point
    b0 = None
    for (x, y), p in np.ndenumerate(img):
        if x == 0 or x == (M-1) or y == 0 or y == (N-1): continue
        if p == 255:
            b0 = Point(x, y)
            break

    # Offset of the 8 neighbors ordered clockwise
    offsets = np.array([
        [+0, -1],
        [-1, -1],
        [-1, +0],
        [-1, +1],
        [+0, +1],
        [+1, +1],
        [+1, +0],
        [+1, -1]
    ]);

    print b0

    # West neighbor of b0
    c0 = b0.neighbor(0)
    print c0

    b1 = None
    c1 = None
    for (x, y) in b0.neighbors(c0):
        print x, y
        if img[x, y] == 255:
            b1 = Point(x, y)
            c1 = previous_c
            break
        previous_c = np.array([x, y])

    print b1, c1

    b = b1
    c = c1

    b1_reached = False

    step = 0

    sequence = []
    sequence.append(b0)
    sequence.append(b)

    while True:

        debug = np.copy(img)
        xx, yy = b.b
        debug[xx, yy] = 8
        xx, yy = c
        debug[xx, yy] = 2
        print debug
        print ""

        for (x, y) in b.neighbors(c):
            if img[x, y] == 255:
                next_b = Point(x, y)
                c = previous_c
                break
            previous_c = np.array([x, y])

        for (x, y) in next_b.neighbors(c):
            if img[x, y] == 255:
                b_tmp = Point(x, y)
                b1_reached = np.array_equal(b_tmp.b, b1.b)
                c_tmp = previous_c
                break
            previous_c = np.array([x, y])

        step += 1

        b = next_b
        sequence.append(b)

        if b1_reached:
            print "B1 REACHED (b = {})".format(b)
            if np.array_equal(b.b, b0.b):
                print "b == b0"
                break

    return sequence
