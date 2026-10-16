# The app icon (1024x1024 RGB PNG): a sphere with its grid and the dots to aim at, on a deep blue ground. Plain Python.
import math, struct, zlib, sys
N, SS = 1024, 2


def mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def color(x, y):
    t = y / N
    c = mix((16, 28, 60), (28, 70, 110), t)
    cx, cy, r = 512, 512, 360
    dx, dy = x - cx, y - cy
    d = math.hypot(dx, dy)
    if d <= r:
        nz = math.sqrt(max(0, 1 - (d / r) ** 2))
        shade = 0.55 + 0.45 * max(0, (-dx * 0.5 - dy * 0.6 + nz * 600) / 800)
        c = mix((40, 120, 200), (150, 220, 250), shade)
        lon = math.atan2(dx, nz * r)
        lat = math.asin(max(-1, min(1, dy / r)))
        for k in range(-2, 3):  # latitude rings
            if abs(lat - k * math.pi / 6) < 0.012 / max(0.25, math.cos(lat)):
                c = mix(c, (255, 255, 255), 0.65)
        for k in range(-3, 4):  # meridians
            if abs(lon - k * math.pi / 6) < 0.012 / max(0.35, math.cos(lat)) and nz > 0.05:
                c = mix(c, (255, 255, 255), 0.55)
        for (alon, alat) in ((0, 0), (math.pi / 6, math.pi / 6), (-math.pi / 6, -math.pi / 6), (math.pi / 3, 0), (-math.pi / 3, 0)):  # the dots
            px, py = cx + r * math.cos(alat) * math.sin(alon), cy + r * math.sin(alat)
            dd = math.hypot(x - px, y - py)
            if 22 <= dd <= 34:
                c = (255, 255, 255)
            elif dd < 22:
                c = mix(c, (255, 255, 255), 0.35)
    return c


rows = []
for y in range(N):
    row = bytearray([0])
    for x in range(N):
        acc = [0.0, 0.0, 0.0]
        for sy in range(SS):
            for sx in range(SS):
                col = color(x + (sx + 0.5) / SS, y + (sy + 0.5) / SS)
                for i in range(3):
                    acc[i] += col[i]
        row += bytes(int(v / (SS * SS) + 0.5) for v in acc)
    rows.append(bytes(row))


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", N, N, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
print("icon", len(png), "bytes")
