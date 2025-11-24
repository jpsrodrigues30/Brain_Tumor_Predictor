import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils
import random

# IMG_SIZE = 256
IMG_SIZE = 224  # MobileNetV2 redimension
EPS = 1e-8      # Avoid division by 0 during zscore normalization
AUG_PER_IMAGE = 3  # How many variables will be generated during augmentation
TARGET_PER_CLASS = 3500 # Target number of images per class after augmentation (only for training set)

def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])

	# add extra margin to the image
	ADD_PIXELS = 0
	y1 = max(extTop[1]  - ADD_PIXELS, 0)
	y2 = min(extBot[1]  + ADD_PIXELS, img.shape[0])
	x1 = max(extLeft[0] - ADD_PIXELS, 0)
	x2 = min(extRight[0] + ADD_PIXELS, img.shape[1])
	new_img = img[y1:y2, x1:x2].copy()
	
	return new_img
	
def apply_clahe(img):
	"""
	Apply CLAHE to improve image contrast

	- Convert to gray scale
	- Apply CLAHE
	- Go 3 channels back to keep shape (H, W, 3)
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Create CLAHE object
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	enhanced = clahe.apply(gray)

	enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
	return enhanced_bgr

def zscore_normalize(img):
	"""
	Receives a uint8 image BGR, shape (H,W,3)
	Makes z-score for each image
	"""
	img_float = img.astype("float32")

	mean = np.mean(img_float)
	std = np.std(img_float)

	img_zscore = (img_float - mean) / (std + EPS) 

	# crop out outliers
	img_zscore = np.clip(img_zscore, -5.0, 5.0)

	# [0,1] conversion
	img_norm01 = (img_zscore + 5.0) / 10.0

	return img_norm01.astype("float32")

def _random_flip(img):
	"""
	Applies horizontal flip to the image (50% chance)
	Expected image format: float32 [0,1]
	"""
	if random.random() < 0.5:
		return np.ascontiguousarray(np.flip(img, axis=1))
	return img

def _random_rotate(img, max_angle=10):
	"""
	Applies slight rotation (-max_angle <-> +max_angle) in degrees.
	Keep fixed size
	"""
	h, w = img.shape[:2]
	angle = random.uniform(-max_angle, +max_angle)
	M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)

	rotated_img = cv2.warpAffine(
		img,
		M,
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT_101
	)

	if rotated_img.ndim == 2:
		rotated_img = np.stack([rotated_img]*3, axis=-1)
	elif rotated_img.shape[-1] != 3:
		rotated_img = rotated_img[..., :3]
	return rotated_img

def _random_intensity_jitter(img, alpha_range=(0.9, 1.1), beta_range=(-0.05, 0.05)):
	"""
	Applies small change to image contrast (alpha) and brightness (beta)
	Expected img in float32 format [0,1]
	"""
	alpha = random.uniform(*alpha_range)
	beta = random.uniform(*beta_range)
	jittered = img * alpha + beta
	jittered = np.clip(jittered, 0.0, 1.0)
	return jittered

def make_augmentations(img_norm, n_aug=AUG_PER_IMAGE):
	"""
	Generates n_aug variations of the entry image
	Each variant will apply a random sequence of:
	-> horizontal flip
	-> slight rotation
	-> jitter of brightness/contrast
	"""
	aug_list = []
	for _ in range(n_aug):
		aug = img_norm.copy()

		ops = [_random_flip, _random_rotate, _random_intensity_jitter]
		random.shuffle(ops)
		for op in ops:
			aug = op(aug)

		aug = np.clip(aug, 0.0, 1.0).astype("float32")

		if aug.ndim == 2:
			aug = np.stack([aug]*3, axis=-1)
		elif aug.shape[-1] != 3:
			aug = aug[..., :3]
		aug_list.append(aug)
	return aug_list


def pre_process(file_path, path_clean, path_normalized, training: bool = False):
	"""
	file_path: Path to the original image file (Example: "dataset/raw/Training")
	path_clean_img: Path used for saving the new image after crop and redimensionalization (Example: "dataset/cleaned/Training)
	path_normalized_img: Path used for saving .npg file generated after normalization (Example: "dataset/normal/Training)
	- Apply image crop
	- Resize image to IMG_SIZExIMG_SIZE
	- Apply CLAHE to improve image contrast (gray)
	- Save claned image (uint8)
	- Apply normalization ([0,1] scale) and salve .npy file for model training	
	"""
	# List the classes files in directory
	classes = os.listdir(file_path)

	for cls in classes: 
		# Get path image file
		cls_src_path = os.path.join(file_path, cls)
		if not os.path.isdir(cls_src_path):
			continue

		cls_path_clean = os.path.join(path_clean, cls)
		cls_path_normalized = os.path.join(path_normalized, cls)
		os.makedirs(cls_path_clean, exist_ok=True)
		os.makedirs(cls_path_normalized, exist_ok=True)

		# List only valid image files in class directory
		image_files = [
			f for f in os.listdir(cls_src_path)
			if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
		]

		n_original = len(image_files)

		# Defines desired total number of images after augmentation
		if training:
			desired_total = max(n_original, TARGET_PER_CLASS)
		else:
			desired_total = n_original  # Avoid oversampling in testing set

		created = 0  # Count how many .npy files were created

		# Loop for every file present in class
		for img_name in tqdm(image_files, desc=f"Processing {file_path}/{cls}"):
			# Create image file path
			img_file_path = os.path.join(cls_src_path, img_name)

			# Read the image
			img = cv2.imread(img_file_path)
			if img is None:
				continue

			# Crop the iamge to only show the brain
			cropped_img = crop_img(img)

			# Image resizing to IMG_SIZExIMG_SIZE
			resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))

			enhanced_img = apply_clahe(resized_img)

			# Saving version uint8 (Allowing visual inspection and debugging)
			img_clean_file_path = os.path.join(cls_path_clean, img_name)
			cv2.imwrite(img_clean_file_path, enhanced_img)

			"""
			Saving normalized version for model training
			Conversion to float32 and using [0,1] scaling
			normalized_img = enhanced_img.astype("float32") / 255.0 # shape (256, 256, 3), values 0.0 - 1.0

			norm_img_name, _ = os.path.splitext(img_name)
			img_normalized_file_path = os.path.join(cls_path_normalized, norm_img_name + ".npy")
			np.save(img_normalized_file_path, normalized_img)
			"""

			# Apply z-scre + rescale to [0,1]
			norm_img = zscore_normalize(enhanced_img)

			# Save new normalized iamge
			norm_img_name, _ = os.path.splitext(img_name)
			img_normalized_file_path = os.path.join(cls_path_normalized, norm_img_name + ".npy")
			np.save(img_normalized_file_path, norm_img)

			created += 1

			# Generates and saves image augmentations
			if training and created<desired_total:
				remaining = desired_total - created
				n_aug_this = min(AUG_PER_IMAGE, remaining)

				aug_list = make_augmentations(norm_img, n_aug=n_aug_this)
				for i, aug_img in enumerate(aug_list):
					aug_out = os.path.join(cls_path_normalized, f"{norm_img_name}_aug{i}.npy")
					np.save(aug_out, aug_img)

				created += n_aug_this

if __name__ == "__main__":
	training_set = "dataset/raw/Training"
	testing_set = "dataset/raw/Testing"

	pre_process(
		file_path=training_set,
		path_clean="dataset/clean/Training",
		path_normalized="dataset/normalized/Training",
		training=True
	)

	pre_process(
		file_path=testing_set,
		path_clean="dataset/clean/Testing",
		path_normalized="dataset/normalized/Testing",
		training=False
	)