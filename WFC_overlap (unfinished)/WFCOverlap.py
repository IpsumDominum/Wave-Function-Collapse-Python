from collections import Counter
from itertools import chain
from random import choice, sample



## still requires bitmap pictures for inputs (8x8 or 16x16 px)




w, h, s = 96, 50, 9
N = 3


def setup():
    size(w * f, h * f, P2D)
    background('#FFFFFF')
    frameRate(1000)
    noStroke()

    global W, A, H, directions, patterns, freqs

    img = loadImage('Flowers.png')
    iw, ih = img.width, img.height
    kernel = tuple(tuple(i + n * iw for i in xrange(N)) for n in xrange(N))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    all = []

    for y in xrange(ih):
        for x in xrange(iw):
            cmat = tuple(
                tuple(img.pixels[((x + n) % iw) + (((a[0] + iw * y) / iw) % ih) * iw] for n in a) for a in kernel)
            for r in xrange(4):
                cmat = zip(*cmat[::-1])
                all.append(cmat)
                all.append(cmat[::-1])
                all.append([a[::-1] for a in cmat])

    all = [tuple(chain.from_iterable(p)) for p in all]
    c = Counter(all)
    freqs = c.values()
    patterns = c.keys()
    npat = len(freqs)

    W = dict(enumerate(tuple(set(range(npat)) for i in xrange(w * h))))
    A = dict(enumerate(tuple(set() for dir in xrange(len(directions))) for i in xrange(npat)))
    H = dict(enumerate(sample(tuple(npat if i > 0 else npat - 1 for i in xrange(w * h)), w * h)))

    for i1 in xrange(npat):
        for i2 in xrange(npat):
            if [n for i, n in enumerate(patterns[i1]) if i % N != (N - 1)] == [n for i, n in enumerate(patterns[i2]) if
                                                                               i % N != 0]:
                A[i1][0].add(i2)
                A[i2][1].add(i1)
            if patterns[i1][:(N * N) - N] == patterns[i2][N:]:
                A[i1][2].add(i2)
                A[i2][3].add(i1)


def draw():
    global H, W

    if not H:
        print
        'finished'
        noLoop()
        return

    emin = min(H, key=H.get)
    id = choice([idP for idP in W[emin] for i in xrange(freqs[idP])])
    W[emin] = {id}
    del H[emin]

    stack = {emin}
    while stack:
        idC = stack.pop()
        for dir, t in enumerate(directions):
            x = (idC % w + t[0]) % w
            y = (idC / w + t[1]) % h
            idN = x + y * w
            if idN in H:
                possible = {n for idP in W[idC] for n in A[idP][dir]}
                if not W[idN].issubset(possible):
                    intersection = possible & W[idN]

                    if not intersection:
                        print
                        'contradiction'
                        noLoop()
                        return

                    W[idN] = intersection
                    H[idN] = len(W[idN]) - random(.1)
                    stack.add(idN)

    fill(patterns[id][0])
    rect((emin % w) * s, (emin / w) * s, s, s)