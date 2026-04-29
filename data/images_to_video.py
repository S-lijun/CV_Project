import os
import imageio

folder = "20260416_211855_905/images"
output = "20260416_211855_905/video.mp4"
fps = 15

files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

with imageio.get_writer(output, fps=fps) as writer:
    for f in files:
        img = imageio.imread(os.path.join(folder, f))
        writer.append_data(img)

print("Video saved:", output)