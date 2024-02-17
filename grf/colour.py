from typing import Any

import numpy as np
from PIL import Image


PIL_PALETTE = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80, 100, 100, 100, 116, 116, 116, 132, 132, 132, 148, 148, 148, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 72, 44, 4, 88, 60, 20, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 64, 0, 4, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 28, 52, 24, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 72, 40, 4, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 40, 20, 112, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 32, 68, 112, 36, 72, 116, 40, 76, 120, 44, 80, 124, 48, 84, 128, 72, 100, 144, 100, 132, 168, 216, 244, 252, 96, 128, 164, 68, 96, 140, 255, 255, 255)
PIL_PALETTE_WIN = (0, 0, 255, 238, 0, 238, 239, 0, 239, 240, 0, 240, 241, 0, 241, 242, 0, 242, 243, 0, 243, 244, 0, 244, 245, 0, 245, 246, 0, 246, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 100, 100, 100, 116, 116, 116, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 132, 132, 132, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 64, 64, 64, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 80, 80, 80, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 48, 48, 48, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 16, 16, 16, 32, 32, 32, 32, 68, 112, 36, 72, 116, 40, 76, 120, 44, 80, 124, 48, 84, 128, 72, 100, 144, 100, 132, 168, 216, 244, 252, 96, 128, 164, 68, 96, 140, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 148, 148, 148, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 255, 255, 255)
PIL_PALETTE_DOS_TOYLAND = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80, 100, 100, 100, 116, 116, 116, 132, 132, 132, 148, 148, 148, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 72, 44, 4, 88, 60, 20, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 64, 0, 4, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 28, 52, 24, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 72, 40, 4, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 40, 20, 112, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 28, 108, 124, 32, 112, 128, 36, 116, 132, 40, 120, 136, 44, 124, 140, 92, 164, 184, 116, 180, 196, 216, 244, 252, 112, 176, 192, 88, 160, 180, 255, 255, 255)
PIL_PALETTE_WIN_TOYLAND = (0, 0, 255, 238, 0, 238, 239, 0, 239, 240, 0, 240, 241, 0, 241, 242, 0, 242, 243, 0, 243, 244, 0, 244, 245, 0, 245, 246, 0, 246, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 100, 100, 100, 116, 116, 116, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 132, 132, 132, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 64, 64, 64, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 80, 80, 80, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 48, 48, 48, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 16, 16, 16, 32, 32, 32, 28, 108, 124, 32, 112, 128, 36, 116, 132, 40, 120, 136, 44, 124, 140, 92, 164, 184, 116, 180, 196, 216, 244, 252, 112, 176, 192, 88, 160, 180, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 148, 148, 148, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 255, 255, 255)
PIL_PALETTE_TTO = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 68, 68, 68, 84, 84, 84, 100, 100, 100, 116, 116, 116, 136, 136, 136, 152, 152, 152, 168, 168, 168, 184, 184, 184, 204, 204, 204, 220, 220, 220, 236, 236, 236, 252, 252, 252, 204, 204, 168, 184, 184, 152, 168, 168, 116, 152, 152, 100, 136, 136, 84, 100, 100, 68, 84, 84, 48, 68, 68, 32, 48, 48, 16, 252, 220, 116, 252, 204, 68, 252, 184, 32, 220, 152, 48, 184, 132, 48, 148, 120, 52, 116, 100, 48, 100, 84, 48, 84, 68, 32, 68, 48, 16, 0, 0, 44, 4, 12, 68, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 132, 196, 236, 168, 216, 244, 204, 236, 252, 48, 0, 0, 80, 0, 0, 108, 4, 4, 136, 12, 12, 164, 20, 20, 196, 32, 32, 224, 44, 44, 252, 56, 56, 84, 32, 0, 100, 48, 0, 132, 64, 0, 168, 80, 0, 200, 96, 0, 236, 152, 0, 236, 184, 68, 252, 236, 136, 124, 140, 60, 116, 128, 48, 104, 116, 36, 96, 104, 28, 84, 92, 20, 76, 80, 12, 68, 72, 4, 56, 60, 0, 48, 48, 0, 252, 60, 0, 252, 96, 0, 252, 128, 0, 252, 164, 0, 252, 196, 0, 48, 16, 0, 68, 32, 0, 100, 48, 16, 116, 68, 32, 136, 84, 32, 168, 100, 48, 184, 116, 48, 220, 136, 68, 236, 152, 84, 236, 168, 100, 252, 184, 116, 252, 204, 136, 108, 92, 92, 132, 88, 88, 148, 80, 80, 164, 64, 64, 40, 16, 92, 64, 48, 136, 88, 64, 172, 112, 84, 212, 140, 100, 252, 160, 140, 252, 204, 192, 180, 184, 172, 152, 164, 148, 132, 144, 132, 112, 120, 100, 76, 96, 72, 48, 68, 48, 24, 44, 28, 12, 56, 44, 12, 56, 64, 32, 64, 48, 0, 100, 68, 28, 84, 68, 0, 116, 184, 160, 88, 160, 124, 64, 136, 92, 44, 116, 60, 24, 92, 32, 12, 72, 12, 0, 52, 0, 116, 116, 84, 168, 152, 100, 220, 204, 136, 252, 236, 168, 116, 100, 84, 152, 136, 116, 204, 184, 168, 252, 220, 204, 200, 132, 108, 192, 120, 96, 176, 100, 80, 168, 88, 68, 148, 68, 52, 132, 48, 36, 116, 32, 20, 100, 28, 12, 68, 12, 0, 32, 16, 0, 112, 140, 24, 80, 124, 16, 56, 108, 12, 36, 96, 4, 40, 84, 16, 36, 72, 0, 36, 64, 0, 136, 96, 60, 124, 84, 44, 112, 68, 32, 100, 56, 20, 88, 44, 12, 72, 36, 4, 56, 24, 0, 16, 16, 48, 32, 32, 68, 48, 48, 100, 68, 72, 120, 92, 96, 140, 120, 120, 164, 152, 152, 184, 168, 168, 204, 180, 196, 116, 168, 184, 84, 148, 168, 84, 128, 156, 80, 112, 140, 80, 96, 124, 76, 72, 100, 60, 52, 76, 48, 32, 48, 32, 200, 116, 32, 176, 100, 24, 156, 84, 16, 132, 72, 12, 112, 56, 4, 88, 40, 0, 68, 28, 0, 252, 252, 180, 252, 100, 100, 148, 204, 168, 180, 208, 128, 236, 188, 148, 228, 172, 132, 220, 160, 120, 212, 144, 108, 204, 128, 96, 196, 116, 84, 184, 104, 72, 168, 96, 64, 148, 84, 52, 128, 76, 40, 112, 64, 32, 92, 52, 20, 180, 124, 112, 172, 104, 100, 164, 88, 84, 132, 72, 68, 104, 48, 48, 80, 32, 32, 241, 0, 241, 242, 0, 242, 243, 0, 243, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 198, 132, 108, 190, 120, 96, 174, 100, 80, 166, 88, 68, 146, 68, 52, 130, 48, 36, 114, 32, 20, 98, 28, 12, 66, 12, 0, 30, 16, 0, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 0, 188, 80, 4, 188, 92, 122, 124, 124, 142, 144, 144, 162, 164, 164, 182, 184, 184, 202, 204, 204, 222, 224, 224, 242, 244, 244, 32, 68, 112, 255, 60, 0, 255, 84, 0, 255, 108, 0, 255, 132, 0, 255, 156, 0, 255, 180, 0, 255, 204, 0, 50, 50, 0, 64, 64, 0, 78, 78, 0, 255, 255, 0, 72, 100, 144, 100, 132, 168, 216, 244, 252, 96, 128, 164, 68, 96, 140, 64, 0, 0, 255, 0, 0, 36, 72, 116, 40, 76, 120, 44, 80, 124, 48, 84, 128, 255, 255, 255)
PIL_PALETTE_TTO_MARS = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 68, 68, 68, 84, 84, 84, 100, 100, 100, 116, 116, 116, 136, 136, 136, 152, 152, 152, 168, 168, 168, 184, 184, 184, 204, 204, 204, 220, 220, 220, 236, 236, 236, 252, 252, 252, 204, 204, 168, 184, 184, 152, 168, 168, 116, 152, 152, 100, 136, 136, 84, 100, 100, 68, 84, 84, 48, 68, 68, 32, 48, 48, 16, 252, 220, 116, 252, 204, 68, 252, 184, 32, 220, 152, 48, 184, 132, 48, 148, 120, 52, 116, 100, 48, 100, 84, 48, 84, 68, 32, 68, 48, 16, 0, 0, 44, 4, 12, 68, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 132, 196, 236, 168, 216, 244, 204, 236, 252, 48, 0, 0, 80, 0, 0, 108, 4, 4, 136, 12, 12, 164, 20, 20, 196, 32, 32, 224, 44, 44, 252, 56, 56, 84, 32, 0, 100, 48, 0, 132, 64, 0, 168, 80, 0, 200, 96, 0, 236, 152, 0, 236, 184, 68, 252, 236, 136, 124, 140, 60, 116, 128, 48, 104, 116, 36, 96, 104, 28, 84, 92, 20, 76, 80, 12, 68, 72, 4, 56, 60, 0, 48, 48, 0, 252, 60, 0, 252, 96, 0, 252, 128, 0, 252, 164, 0, 252, 196, 0, 48, 16, 0, 68, 32, 0, 100, 48, 16, 116, 68, 32, 136, 84, 32, 168, 100, 48, 184, 116, 48, 220, 136, 68, 236, 152, 84, 236, 168, 100, 252, 184, 116, 252, 204, 136, 108, 92, 92, 132, 88, 88, 148, 80, 80, 164, 64, 64, 40, 16, 92, 64, 48, 136, 88, 64, 172, 112, 84, 212, 140, 100, 252, 160, 140, 252, 204, 192, 180, 184, 172, 152, 164, 148, 132, 144, 132, 112, 120, 100, 76, 96, 72, 48, 68, 48, 24, 44, 28, 12, 56, 44, 12, 56, 64, 32, 64, 48, 0, 100, 68, 28, 84, 68, 0, 116, 184, 160, 88, 160, 124, 64, 136, 92, 44, 116, 60, 24, 92, 32, 12, 72, 12, 0, 52, 0, 116, 116, 84, 168, 152, 100, 220, 204, 136, 252, 236, 168, 116, 100, 84, 152, 136, 116, 204, 184, 168, 252, 220, 204, 200, 132, 108, 192, 120, 96, 176, 100, 80, 168, 88, 68, 148, 68, 52, 132, 48, 36, 116, 32, 20, 100, 28, 12, 68, 12, 0, 32, 16, 0, 112, 140, 24, 80, 124, 16, 56, 108, 12, 36, 96, 4, 40, 84, 16, 36, 72, 0, 36, 64, 0, 136, 96, 60, 124, 84, 44, 112, 68, 32, 100, 56, 20, 88, 44, 12, 72, 36, 4, 56, 24, 0, 16, 16, 48, 32, 32, 68, 48, 48, 100, 68, 72, 120, 92, 96, 140, 120, 120, 164, 152, 152, 184, 168, 168, 204, 180, 196, 116, 168, 184, 84, 148, 168, 84, 128, 156, 80, 112, 140, 80, 96, 124, 76, 72, 100, 60, 52, 76, 48, 32, 48, 32, 200, 116, 32, 176, 100, 24, 156, 84, 16, 132, 72, 12, 112, 56, 4, 88, 40, 0, 68, 28, 0, 252, 252, 180, 252, 100, 100, 148, 204, 168, 180, 208, 128, 236, 188, 148, 228, 172, 132, 220, 160, 120, 212, 144, 108, 204, 128, 96, 196, 116, 84, 184, 104, 72, 168, 96, 64, 148, 84, 52, 128, 76, 40, 112, 64, 32, 92, 52, 20, 180, 124, 112, 172, 104, 100, 164, 88, 84, 132, 72, 68, 104, 48, 48, 80, 32, 32, 241, 0, 241, 242, 0, 242, 243, 0, 243, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 198, 132, 108, 190, 120, 96, 174, 100, 80, 166, 88, 68, 146, 68, 52, 130, 48, 36, 114, 32, 20, 98, 28, 12, 66, 12, 0, 30, 16, 0, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 0, 188, 80, 4, 188, 92, 122, 124, 124, 142, 144, 144, 162, 164, 164, 182, 184, 184, 202, 204, 204, 222, 224, 224, 242, 244, 244, 116, 0, 0, 255, 60, 0, 255, 84, 0, 255, 108, 0, 255, 132, 0, 255, 156, 0, 255, 180, 0, 255, 204, 0, 50, 50, 0, 64, 64, 0, 78, 78, 0, 255, 255, 0, 156, 40, 12, 200, 76, 68, 228, 112, 108, 208, 84, 76, 164, 48, 20, 64, 0, 0, 255, 0, 0, 124, 4, 0, 136, 16, 4, 148, 28, 8, 160, 44, 16, 255, 255, 255)

PALETTE = [tuple(PIL_PALETTE[3 * i: 3 * i + 3]) for i in range(256)]
NP_PALETTE = np.array(PALETTE, dtype=np.uint8)
SAFE_COLOURS = tuple(range(1, 0xD7))
ALL_COLOURS = tuple(range(256))
WATER_COLOURS = tuple(range(0xF5, 0xFF))
DEFAULT_BRIGHTNESS = 128  # RGB brightness used to pass masked colour without adjustment


CC_COLOURS = [
    [198 , 199 , 200 , 201 , 202 , 203 , 204 , 205 ],
    [96 , 97 , 98 , 99 , 100 , 101 , 102 , 103 ],
    [42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 ],
    [62 , 63 , 64 , 65 , 66 , 67 , 68 , 69 ],
    [179 , 180 , 181 , 182 , 183 , 164 , 165 , 166 ],
    [154 , 155 , 156 , 157 , 158 , 159 , 160 , 161 ],
    [82 , 83 , 84 , 85 , 206 , 207 , 208 , 209 ],
    [88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 ],
    [146 , 147 , 148 , 149 , 150 , 151 , 152 , 153 ],
    [114 , 115 , 116 , 117 , 118 , 119 , 120 , 121 ],
    [128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 ],
    [136 , 137 , 138 , 139 , 140 , 141 , 142 , 143 ],
    [64 , 192 , 193 , 194 , 195 , 196 , 197 , 39 ],
    [32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 ],
    [4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 ],
    [8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 ]
]

WIN_TO_DOS = [
      0,   1,   2,   3,   4,   5,   6,   7,
      8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,
     24,  25,  26,  27,  28,  29,  30,  31,
      6,   7,  34,  35,  36,  37,  38,  39,
      8,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,
     56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,
     72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,
      4,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103,
    104, 105,   5, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135,
      3, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214,   1,
      2, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 227, 228, 229, 230, 231,
    232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244,   9, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 255,
]

LAZY_CONSTANT_GENERATORS = {
    'OKLAB_PALETTE': lambda: np.array([srgb_to_oklab(c) for c in PALETTE]),
    'NP_PALETTE': lambda: np.array(PALETTE),
    'PALETTE_IDX': lambda: {p: i for i, p in enumerate(PALETTE)},
}
_LAZY_CONSTANTS = {}


def __getattr__(name: str) -> Any:
    func = LAZY_CONSTANT_GENERATORS.get(name)
    if func is not None:
        value = _LAZY_CONSTANTS.get(name)
        if value is None:
            value = func()
            _LAZY_CONSTANTS[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def srgb_to_linear(rgb):
    rgb = np.array(rgb) / 255.
    mask = rgb <= 0.04045
    rgb[mask] /= 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    return rgb


def linear_to_srgb(rgb_linear):
    mask = rgb_linear <= 0.0031308
    rgb_linear[mask] *= 12.92
    rgb_linear[~mask] = 1.055 * (rgb_linear[~mask] ** (1 / 2.4)) - 0.055
    return np.rint(np.clip(rgb_linear * 255., 0, 255.))


LRGB_TO_OKLAB_M1 = np.array((
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883029595, 0.2817188376, 0.6299787005),
))
LRGB_TO_OKLAB_M2 = np.array((
    (0.2104542553, +0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, +0.4505937099),
    (0.0259040371, +0.7827717662, -0.8086757660),
))


def srgb_to_oklab(rgb):
    rgb_linear = srgb_to_linear(rgb)
    x = LRGB_TO_OKLAB_M1.dot(rgb_linear)
    return LRGB_TO_OKLAB_M2.dot(np.cbrt(x))
    # TOOD mass convert ?
    # x = LRGB_TO_OKLAB_M1 @ rgb_linear.T
    # return (LRGB_TO_OKLAB_M2 @ np.cbrt(x)).T


OKLAB_TO_LRGB_M1 = np.array((
    (        1.,          1.,          1.),
    (0.39633778, -0.10556135, -0.08948418),
    (0.21580376, -0.06385417, -1.29148555),
))
OKLAB_TO_LRGB_M2 = np.array((
    ( 4.07674166, -1.2684380, -0.00419609),
    (-3.30771159,  2.6097574, -0.70341861),
    ( 0.23096993, -0.3413194,  1.70761470),
))


def oklab_to_srgb(lab):
    x = lab.dot(OKLAB_TO_LRGB_M1)
    rgb = (x ** 3).dot(OKLAB_TO_LRGB_M2)
    return linear_to_srgb(rgb)


def oklab_find_best_colour(x, in_range=SAFE_COLOURS):
    colours = __getattr__('OKLAB_PALETTE').take(in_range, axis=0)
    if len(x.shape) == 1:
        return in_range[np.argmin(np.sum((colours - x) ** 2, axis=1))]
    xi, ci = np.indices((len(x), len(colours)))
    return [in_range[x] for x in np.argmin(np.sum((x[xi] - colours[ci]) ** 2, axis=2), axis=1)]


def oklab_blend(source, tint, ratio=0.5):
    return source * (1 - ratio) + tint * ratio


def srgb_find_best_colour(x, in_range=SAFE_COLOURS):
    return oklab_find_best_colour(srgb_to_oklab(x), in_range=in_range)


def make_palette_image(palette, size=20):
    npimg = np.zeros((16 * size, 16 * size, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        x = (i % 16) * size
        y = (i // 16) * size
        npimg[y:y + size, x:x + size] = c
    return Image.fromarray(npimg)


def srgb_color_distance(c1, c2):
    rmean = (c1.rgb[0] + c2.rgb[0]) / 2.
    r = c1.rgb[0] - c2.rgb[0]
    g = c1.rgb[1] - c2.rgb[1]
    b = c1.rgb[2] - c2.rgb[2]
    return math.sqrt(
        ((2 + rmean) * r * r) +
        4 * g * g +
        (3 - rmean) * b * b)


def openttd_adjust_brightness(c, brightness):
    if brightness == 128:
        return c

    r, g, b = c
    combined = (r << 32) | (g << 16) | b
    combined *= brightness

    r = (combined >> 39) & 0x1ff
    g = (combined >> 23) & 0x1ff
    b = (combined >> 7) & 0x1ff

    if (combined & 0x800080008000) == 0:
        return (r, g, b)

    ob = 0
    # Sum overbright
    if r > 255: ob += r - 255
    if g > 255: ob += g - 255
    if b > 255: ob += b - 255

    # Reduce overbright strength
    ob //= 2
    return (
        255 if r >= 255 else min(r + ob * (255 - r) // 256, 255),
        255 if g >= 255 else min(g + ob * (255 - g) // 256, 255),
        255 if b >= 255 else min(b + ob * (255 - b) // 256, 255),
    )
