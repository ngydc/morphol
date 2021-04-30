import morphol
from PIL import Image
import numpy as np


def main():
    img_path = "./images/building.jpg"
    image = Image.open(img_path).convert('L')
    img_mat = np.array(image, dtype=np.float64)
    eroded = morphol.erode(img_mat, 40)
    eroded_img = Image.fromarray(np.uint8(eroded), 'L')
    eroded_img.save(img_path.replace('.jpg', '_eroded.jpg'), format='JPEG')

    dilated = morphol.dilate(img_mat, 40)
    dilated_img = Image.fromarray(np.uint8(dilated), 'L')
    dilated_img.save(img_path.replace('.jpg', '_dilated.jpg'), format='JPEG')


if __name__ == '__main__':
    main()
