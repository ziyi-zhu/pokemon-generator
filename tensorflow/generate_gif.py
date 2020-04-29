import glob
import imageio
import PIL

# Display a single image using the epoch number
def display_image(epoch_no):
	return PIL.Image.open('./generated/image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
	filenames = glob.glob('./generated/image*.png')
	filenames = sorted(filenames)
	last = -1
	for i,filename in enumerate(filenames):
		frame = 2*(i**0.5)
		if round(frame) > round(last):
			last = frame
		else:
			continue
		image = imageio.imread(filename)
		writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)