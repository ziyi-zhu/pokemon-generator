from PIL import Image

import os

image_dir = './icons'
data_dir = './dataset'

for i, f in enumerate(os.listdir(image_dir)):
	path = os.path.join(image_dir, f)
	img = Image.open(path)

	img = img.convert('RGBA')

	background = Image.new('RGB', (28, 28), 'white')
	background.paste(img, (-6, -1), img.split()[-1])
	img = background

	img = img.save(os.path.join(data_dir, '{:03d}.png'.format(i + 1)))

	print('Processing image {:03d}'.format(i + 1))