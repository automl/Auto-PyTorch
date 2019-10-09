import numpy as np
import math
import random
import os

from PIL import Image, ImageOps, ImageEnhance

class Operation(object):
    """
    Base class of all operations.
    """
    def __init__(self, prob, magnitude):
        self.prob = prob
        self.magnitude = magnitude

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, image):
        raise NotImplementedError("Need to instantiate a subclass of this class!")

class Equalize(Operation):
    """
    Equalize the image histogram.
    """
    def __init__(self, prob, magnitude):
        super(Equalize, self).__init__(prob, None)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            return ImageOps.equalize(image)

class Invert(Operation):
    """
    Invert the pixels of the image.
    """
    def __init__(self, prob, magnitude):
        super(Invert, self).__init__(prob, None)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            return ImageOps.invert(image)

class AutoContrast(Operation):
    """
    Maximize the image contrast, by making the darkest pixel black and
    the lightest pixel white.
    """
    def __init__(self, prob, magnitude):
        super(AutoContrast, self).__init__(prob, None)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            return ImageOps.autocontrast(image)

class Posterize(Operation):
    """
    Reduce the number of bits for each pixel magnitude bits.
    """
    def __init__(self, prob, magnitude):
        super(Posterize, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(4, 8, 10)
            bits = int(round(magnitude_range[self.magnitude]))
            return ImageOps.posterize(image, bits)

class Solarize(Operation):
    """
    Invert all pixels above a threshold value of magnitude.
    """
    def __init__(self, prob, magnitude):
        super(Solarize, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(0, 256, 10)
            threshold = magnitude_range[self.magnitude]
            return ImageOps.solarize(image, threshold)

class Contrast(Operation):
    """
    Control the contrast of the image. 
    A magnitude=0 gives a gray image,
    whereas magnitude=1 gives the original image.
    """
    def __init__(self, prob, magnitude):
        super(Contrast, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(0.1, 1.9, 10)
            factor = magnitude_range[self.magnitude]
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)

class Color(Operation):
    """
    Adjust the color balance of the image, 
    in a manner similar to the controls on a colour TV set.
    A magnitude=0 gives a black & white image, 
    whereas magnitude=1 gives the original image.
    """
    def __init__(self, prob, magnitude):
        super(Color, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(0.1, 1.9, 10)
            factor = magnitude_range[self.magnitude]
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)

class Brightness(Operation):
    """
    Adjust the brightness of the image. 
    A magnitude=0 gives a black image,
    whereas magnitude=1 gives the original image.
    """
    def __init__(self, prob, magnitude):
        super(Brightness, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(0.1, 1.9, 10)
            factor = magnitude_range[self.magnitude]
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)

class Sharpness(Operation):
    """
    Adjust the sharpness of the image. 
    A magnitude=0 gives a blurred image,
    whereas magnitude=1 gives the original image.
    """
    def __init__(self, prob, magnitude):
        super(Sharpness, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(0.1, 1.9, 10)
            factor = magnitude_range[self.magnitude]
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)

class Rotate(Operation):
    """
    Rotate the image magnitude degrees.
    """
    def __init(self, prob, magnitude):
        super(Rotate, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(-30, 30, 10)
            degrees = magnitude_range[self.magnitude]
            return image.rotate(degrees, expand=False, resample=Image.BICUBIC)

class TranslateX(Operation):
    """
    Translate the image in the horizontal axis 
    direction by magnitude number of pixels.
    """
    def __init__(self, prob, magnitude):
        super(TranslateX, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(-15, 15, 10)
            pixels = magnitude_range[self.magnitude]
            return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))

class TranslateY(Operation):
    """
    Translate the image in the vertical axis 
    direction by magnitude number of pixels.
    """
    def __init__(self, prob, magnitude):
        super(TranslateY, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(-15, 15, 10)
            pixels = magnitude_range[self.magnitude]
            return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))


class ShearX(Operation):
    """
    Shear image along horizontal axis with rate magnitude.
    """
    def __init__(self, prob, magnitude):
        super(ShearX, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(-0.3, 0.3, 10)
            rate = magnitude_range[self.magnitude]

            w, h = image.size

            phi = math.tan(abs(rate))
            shift_in_pixels = phi * h
            matrix_offset = shift_in_pixels
            if rate <= 0:
                matrix_offset = 0
                phi = -1 * phi

            transform_matrix = (1, phi, -matrix_offset, 0, 1, 0)

            image = image.transform((int(round(w + shift_in_pixels)), h),
                                    Image.AFFINE,
                                    transform_matrix)

            if rate <= 0:
                image = image.crop((0, 0, w, h))
            else:
                image = image.crop((abs(shift_in_pixels), 0, w + abs(shift_in_pixels), h))

            return image

class ShearY(Operation):
    """
    Shear image along vertical axis with rate magnitude.
    """
    def __init__(self, prob, magnitude):
        super(ShearY, self).__init__(prob, magnitude)

    def __call__(self, image):
        if random.uniform(0, 1) > self.prob:
            return image
        else:
            magnitude_range = np.linspace(-0.3, 0.3, 10)
            rate = magnitude_range[self.magnitude]

            w, h = image.size

            phi = math.tan(abs(rate))
            shift_in_pixels = phi * h
            matrix_offset = shift_in_pixels
            if rate <= 0:
                matrix_offset = 0
                phi = -1 * phi

            transform_matrix = (1, 0, 0, phi, 1, -matrix_offset)

            image = image.transform((w, int(round(h + shift_in_pixels))),
                                    Image.AFFINE,
                                    transform_matrix)

            if rate <= 0:
                image = image.crop((0, 0, w, h))
            else:
                image = image.crop((0, abs(shift_in_pixels), w, h + abs(shift_in_pixels)))

            return image
