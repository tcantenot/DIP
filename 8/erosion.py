import numpy as np

from scipy import misc

# Perform an erosion of the input with the given structuring element
def erosion(input, structure, border_value=0, mask=None):

    w, h = input.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = struct_w / 2, struct_h / 2

    output = np.empty(input.shape, dtype=int)

    def get(x, y):
        return border_value if x < 0 or x >= w or y < 0 or y >= h else input[x, y]

    for x in xrange(w):
        for y in xrange(h):
            if mask is not None and not mask[x, y]: continue

            l = float('inf')

            for i in xrange(struct_w):
                for j in xrange(struct_h):
                    if structure[i, j] == 1:
                        l = min(get(x+i-struct_w2, y+j-struct_h2), l)
                        if l == 0: break

            output[x, y] = l

    return output

    #return np.array([input[i[0], i[1]] if np.array_equal(input[i[0]-(structure.shape[0]/2):i[0]+(structure.shape[0]/2)+1, i[1]-(structure.shape[1]/2):i[1]+(structure.shape[1]/2)+1] - structure, np.zeros(structure.shape)) else 0 for i, d in np.ndenumerate(input) if i[0] >= (structure.shape[0]/2) and i[0] < input.shape[0]-(structure.shape[0]/2) and i[1] >= (structure.shape[1]/2) and i[1] < input.shape[1]-(structure.shape[1]/2)]).reshape((input.shape[0]-structure.shape[0]+1, input.shape[1]-structure.shape[1]+1))

def erosion2(input, structure, border_value=1, mask=None):

    w, h = input.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = (struct_w-1) / 2, (struct_h-1) / 2

    output = np.empty(input.shape, dtype=int)

    def get(x, y):
        return border_value if x < 0 or x >= w or y < 0 or y >= h else input[x, y]

    for x in xrange(w):
        for y in xrange(h):

            #if mask is not None and not mask[x, y]:
                #output[x, y] = input[x, y]
                #continue

            l = 2

            for i in xrange(struct_w):
                if l == 0: break
                for j in xrange(struct_h):
                    if structure[i, j] == 1:
                        l = min(get(x+i-struct_w2, y+j-struct_h2), l)
                        if l == 0: break

            output[x, y] = l

    return output

def erosion3(input, structure, border_value=1, mask=None):

    w, h = input.shape
    struct_w, struct_h = structure.shape
    struct_w2, struct_h2 = struct_w / 2, struct_h / 2

    output = np.zeros(input.shape, dtype=bool)

    for x in xrange(struct_w2, w - struct_w2):
        for y in xrange(struct_h2, h - struct_h2):

            #if mask is not None and not mask[x, y]:
                #output[x, y] = input[x, y]
                #continue

            a = input[x-struct_w2:x+struct_w2+1, y-struct_h2:y+struct_h2+1] & structure
            if (a == structure).any():
                output[x, y] = 1

    misc.imsave('erosion.tif', output)
    return output

