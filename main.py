import morphol
from PIL import Image
import numpy as np


def main():
    img_path = "./images/building.jpg"
    image = Image.open(img_path).convert('L')
    img_mat = np.array(image, dtype=np.float64)

    """ standard morphological filters """

    eroded = morphol.erode(img_mat, 20)
    eroded_img = Image.fromarray(np.uint8(eroded), 'L')
    eroded_img.save(img_path.replace('.jpg', '_eroded.jpg'), format='JPEG')

    dilated = morphol.dilate(img_mat, 20)
    dilated_img = Image.fromarray(np.uint8(dilated), 'L')
    dilated_img.save(img_path.replace('.jpg', '_dilated.jpg'), format='JPEG')

    closed = morphol.close(img_mat, 20)
    closed_img = Image.fromarray(np.uint8(closed), 'L')
    closed_img.save(img_path.replace('.jpg', '_closed.jpg'), format='JPEG')

    opened = morphol.open(img_mat, 20)
    opened_img = Image.fromarray(np.uint8(opened), 'L')
    opened_img.save(img_path.replace('.jpg', '_opened.jpg'), format='JPEG')

    """ morphological filters with random window sizes"""
    """
    eroded_rand = morphol.erode_random(img_mat, 10, 20)
    eroded_img_rand = Image.fromarray(np.uint8(eroded_rand), 'L')
    eroded_img_rand.save(img_path.replace('.jpg', '_eroded_rand.jpg'), format='JPEG')

    dilated_rand = morphol.dilate_random(img_mat, 10, 20)
    dilated_img_rand = Image.fromarray(np.uint8(dilated_rand), 'L')
    dilated_img_rand.save(img_path.replace('.jpg', '_dilated_rand.jpg'), format='JPEG')

    opened_rand = morphol.open_random(img_mat, 10, 20)
    opened_img_rand = Image.fromarray(np.uint8(opened_rand), 'L')
    opened_img_rand.save(img_path.replace('.jpg', '_opened_rand.jpg'), format='JPEG')

    closed_rand = morphol.close_random(img_mat, 10, 20)
    closed_img_rand = Image.fromarray(np.uint8(closed_rand), 'L')
    closed_img_rand.save(img_path.replace('.jpg', '_closed_rand.jpg'), format='JPEG')
    """


if __name__ == '__main__':
    main()
