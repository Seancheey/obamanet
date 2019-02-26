# Get keypoint data from the images and prepare for pix2pix
from utils import *

inputFolder = 'images/00034-001/'
outputFolderKp = 'a2key_data/'
saveFilename = outputFolderKp + 'kp_test.pickle'
outputFolderForpix2pix = 'pix2pix_input/'
inputToA2KeyModel = 'a2key_data/images/'

cmd = 'rm -rf ' + inputFolder + '*-square-x-100.jpeg'
subprocess.call(cmd, shell=True)

if not(os.path.exists(outputFolderForpix2pix)):
	# Create directory
	subprocess.call('mkdir -p ' + outputFolderForpix2pix, shell=True)
if not(os.path.exists(outputFolderKp)):
	# Create directory
	subprocess.call('mkdir -p ' + outputFolderKp, shell=True)
if not(os.path.exists(inputToA2KeyModel)):
	# Create directory
	subprocess.call('mkdir -p ' + inputToA2KeyModel, shell=True)

searchNames = inputFolder + '*' + '.jpeg'
filenames = sorted(glob(searchNames))

d = []
for file in tqdm(filenames):
	img = cv2.imread(file)
	x = int(np.floor((img.shape[1]-256)/2))

	# Crop to a square image
	crop_img = img[0:256, x:x+256]
	outputName = file[0:-len('.jpeg')]+'-square-x-100.jpeg'
	# print(outputName)
	cv2.imwrite(outputName, crop_img)

	# extract the keypoints
	keypoints = get_facial_landmarks(outputName)
	l = getKeypointFeatures(keypoints)
	unit_kp, N, tilt, mean = l[0], l[1], l[2], l[3]
	kp_mouth = unit_kp[48:68]
	store_list = [kp_mouth, N, tilt, mean, unit_kp, keypoints]
	d.append(store_list)

	# create a patch based on the tilt, mean and the size of face
	mean_x, mean_y = int(mean[0]), int(mean[1])
	size = int(N/15)
	aspect_ratio_mouth = 1.8
	# print('mean (y, x):', mean_y, mean_x, 'size:', size)

	patch_img = crop_img.copy()
	# patch = np.zeros_like(patch_img[ mean_y-size: mean_y+size, mean_x-size: mean_x+size ])
	patch_img[ mean_y-size: mean_y+size, mean_x-int(aspect_ratio_mouth*size):
		mean_x+int(aspect_ratio_mouth*size) ] = 0
	cv2.imwrite(inputToA2KeyModel + file[len(inputFolder):-len('.jpeg')] + '.png', patch_img)


	drawLips(keypoints, patch_img)

	# Slap the other original image onto this
	patch_img = np.hstack((patch_img, crop_img))

	outputNamePatch = outputFolderForpix2pix + file[len(inputFolder):-len('.jpeg')] + '.png'
	cv2.imwrite(outputNamePatch, patch_img)

# save the extracted keypoints
with open(saveFilename, "wb") as output_file:
	pkl.dump(d, output_file)

	

