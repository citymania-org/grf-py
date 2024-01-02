import numpy as np
from PIL import Image
import random

import grf


PADDING = 10
W = 65
H = 20
I = 4

n = len(grf.CC_COLOURS)

w = (W + PADDING) * n + PADDING
h = (H + PADDING) * n + PADDING
rgb = np.zeros((h, w, 3), dtype=np.uint8)

for i, il in enumerate(grf.CC_COLOURS):
	for j, jl in enumerate(grf.CC_COLOURS):
		x = i * (W + PADDING) + PADDING
		y = j * (H + PADDING) + PADDING

		# full colour gradient
		# ic, jc = grf.OKLAB_PALETTE[il[I]], grf.OKLAB_PALETTE[jl[I]]
		# for k in range(W + 1):
		# 	co = grf.oklab_blend(ic, jc, ratio=k / W)
		# 	cs = grf.oklab_to_srgb(co)
		# 	rgb[y: y + H, x + k] = cs

		# two colour gradient
		ic, jc = grf.PALETTE[il[I]], grf.PALETTE[jl[I]]
		for k in range(W + 1):
			rgb[y:y + H, x + k] = ic
			random.seed(k)
			f = random.randrange(101) / 100
			s = k / W
			for ii in range(H):
				f += s
				if f > 0.99:
					rgb[y + ii, x +k] = jc
					f -= 1

im = Image.fromarray(rgb)
im.show()
