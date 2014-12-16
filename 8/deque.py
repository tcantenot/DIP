from collections import deque

class MinDeque(object):

    def __init__(self):
        self._deque = deque()
        self._min = deque()

    def push(self, value):
        self._deque.append(value)
        while len(self._min) > 0 and self._min[-1] > value:
            self._min.pop()
        self._min.append(value)

    def pop(self):
        if self._deque[0] == self._min[0]:
            self._min.popleft()
        return self._deque.popleft()

    def empty(self):
        while len(self._deque) > 0:
            self._deque.pop()
        while len(self._min) > 0:
            self._min.pop()


    def min(self):
        return self._min[0]


class MaxDeque(object):

    def __init__(self):
        self._deque = deque()
        self._max = deque()

    def push(self, value):
        self._deque.append(value)
        while len(self._max) > 0 and self._max[-1] < value:
            self._max.pop()
        self._max.append(value)

    def pop(self):
        if self._deque[0] == self._max[0]:
            self._max.popleft()
        return self._deque.popleft()

    def empty(self):
        while len(self._deque) > 0:
            self._deque.pop()
        while len(self._max) > 0:
            self._max.pop()

    def max(self):
        return self._max[0]


if __name__ == '__main__':

    print "### MinQueue ###\n"

    mid = MinDeque()

    mid.push(12)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.push(5)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.push(10)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.push(7)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.push(11)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.push(19)
    print mid._deque, mid._min, mid.min()
    print ""

    mid.pop()
    print mid._deque, mid._min, mid.min()
    print ""

    mid.pop()
    print mid._deque, mid._min, mid.min()
    print ""

    mid.pop()
    print mid._deque, mid._min, mid.min()
    print ""

    mid.pop()
    print mid._deque, mid._min, mid.min()
    print ""

    mid.pop()
    print mid._deque, mid._min, mid.min()
    print ""


    print "\n### MaxQueue ###\n"

    mad = MaxDeque()

    mad.push(12)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.push(5)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.push(19)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.push(7)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.push(11)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.push(10)
    print mad._deque, mad._max, mad.max()
    print ""

    mad.pop()
    print mad._deque, mad._max, mad.max()
    print ""

    mad.pop()
    print mad._deque, mad._max, mad.max()
    print ""

    mad.pop()
    print mad._deque, mad._max, mad.max()
    print ""

    mad.pop()
    print mad._deque, mad._max, mad.max()
    print ""

    mad.pop()
    print mad._deque, mad._max, mad.max()
    print ""


    mad.empty()
    print mad._deque, mad._max
    print ""
    mad.push(1)
    print mad._deque, mad._max, mad.max()
    print ""
    mad.push(0)
    print mad._deque, mad._max, mad.max()
    print ""
    mad.push(0)
    print mad._deque, mad._max, mad.max()
    print ""
    mad.push(0)
    print mad._deque, mad._max, mad.max()
    print ""
    mad.pop()
