import numpy as np

def erosion(input, structure, border_value=0, mask=None):

    w, h = input.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = (struct_w-1) / 2, (struct_h-1) / 2

    output = np.empty(input.shape, dtype=int)

    def get(x, y):
        return border_value if x < 0 or x >= w or y < 0 or y >= h else input[x, y]

    for x in xrange(w):
        for y in xrange(h):

            if mask is not None and not mask[x, y]:
                output[x, y] = input[x, y]
                continue

            l = 99999999
            for i in xrange(struct_w):
                if l == 0: break
                for j in xrange(struct_h):
                    if structure[i, j] == 1:
                        l = min(get(x+i-struct_w2, y+j-struct_h2), l)
                        if l == 0: break

            output[x, y] = l

    return output
