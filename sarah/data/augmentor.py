import cv2
import numpy as np


class Augmentor():
    """
    Image augmentation class for applying various transformations to images.
    """

    def __init__(self,
                 binarize=None,
                 erode=None,
                 dilate=None,
                 elastic=None,
                 perspective=None,
                 mixup=None,
                 shear=None,
                 scale=None,
                 rotate=None,
                 shift_y=None,
                 shift_x=None,
                 salt_and_pepper=None,
                 gaussian_noise=None,
                 gaussian_blur=None,
                 seed=None):
        """
        Initializes a new instance of the Augmentor class.

        Parameters
        ----------
        binarize : list or None, optional
            Parameters for binarization.
        erode : list or None, optional
            Parameters for erode transformation.
        dilate : list or None, optional
            Parameters for dilate transformation.
        elastic : list or None, optional
            Parameters for elastic transformation.
        perspective : list or None, optional
            Parameters for perspective transform transformation.
        mixup : list or None, optional
            Parameters for mixup transformation.
        shear : list or None, optional
            Parameters for shear transformation.
        scale : list or None, optional
            Parameters for scale transformation.
        rotate : list or None, optional
            Parameters for rotate transformation.
        shift_y : list or None, optional
            Parameters for vertical translation transformation.
        shift_x : list or None, optional
            Parameters for horizontal translation transformation.
        salt_and_pepper : list or None, optional
            Parameters for salt and pepper noise.
        gaussian_noise : list or None, optional
            Parameters for gaussian noise.
        gaussian_blur : list or None, optional
            Parameters for Gaussian blur transformation.
        seed : int or None, optional
            Seed for random values from numpy.
        """

        seed = seed or 0
        np.random.seed(seed)

        self.binarize_params = binarize
        self.erode_params = erode
        self.dilate_params = dilate
        self.elastic_params = elastic
        self.perspective_params = perspective
        self.mixup_params = mixup
        self.shear_params = shear
        self.scale_params = scale
        self.rotate_params = rotate
        self.shift_y_params = shift_y
        self.shift_x_params = shift_x
        self.salt_and_pepper_params = salt_and_pepper
        self.gaussian_noise_params = gaussian_noise
        self.gaussian_blur_params = gaussian_blur
        self.seed = seed

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        pad, width = 25, 60
        info = "=" * width
        info += f"\n{self.__class__.__name__.center(width)}"
        info += "\n" + "-" * width
        info += f"\n{'binarize':<{pad}}: {self.binarize_params or '-'}"
        info += f"\n{'erode':<{pad}}: {self.erode_params or '-'}"
        info += f"\n{'dilate':<{pad}}: {self.dilate_params or '-'}"
        info += f"\n{'elastic':<{pad}}: {self.elastic_params or '-'}"
        info += f"\n{'perspective':<{pad}}: {self.perspective_params or '-'}"
        info += f"\n{'mixup':<{pad}}: {self.mixup_params or '-'}"
        info += f"\n{'shear':<{pad}}: {self.shear_params or '-'}"
        info += f"\n{'scale':<{pad}}: {self.scale_params or '-'}"
        info += f"\n{'rotate':<{pad}}: {self.rotate_params or '-'}"
        info += f"\n{'shift_y':<{pad}}: {self.shift_y_params or '-'}"
        info += f"\n{'shift_x':<{pad}}: {self.shift_x_params or '-'}"
        info += f"\n{'salt_and_pepper':<{pad}}: {self.salt_and_pepper_params or '-'}"
        info += f"\n{'gaussian_noise':<{pad}}: {self.gaussian_noise_params or '-'}"
        info += f"\n{'gaussian_blur':<{pad}}: {self.gaussian_blur_params or '-'}"
        info += f"\n{'seed':<{pad}}: {self.seed}"

        return info

    def augmentation(self, image, batch_images=None):
        """
        Apply transformations to an image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        batch_images : list
            List of images used for mixup transformation.

        Returns
        -------
        ndarray
            Transformed image.
        """

        mixup_params = self.mixup_params if self.mixup_params is None else \
            self.mixup_params[:1] + [batch_images] + self.mixup_params[1:]

        transformations = [
            (self.binarize, self.binarize_params, None),
            (self.erode, self.erode_params, 32),
            (self.dilate, self.dilate_params, 32),
            (self.elastic, self.elastic_params, 32),
            (self.perspective, self.perspective_params, 32),
            (self.mixup, mixup_params, 32),
            (self.shear, self.shear_params, 32),
            (self.scale, self.scale_params, 32),
            (self.rotate, self.rotate_params, None),
            (self.shift_y, self.shift_y_params, None),
            (self.shift_x, self.shift_x_params, None),
            (self.salt_and_pepper, self.salt_and_pepper_params, None),
            (self.gaussian_noise, self.gaussian_noise_params, 32),
            (self.gaussian_blur, self.gaussian_blur_params, 32),
        ]

        for transform_func, params, min_size in transformations:
            if params is None or len(params) == 0:
                continue

            if min_size and min(image.shape[:2]) < min_size:
                continue

            if np.random.random() <= float(params[0]):
                image = transform_func(image, *params[1:])

        return image

    def binarize(self, image, method='otsu'):
        """
        Apply binarization method to an image.

        Parameters
        ----------
        image : ndarray
            Input image to be binarized.
        method : str, optional
            Binarization method to apply.

        Returns
        ----------
        ndarray
            Binarized image.
        """

        if method == 'global':
            _, image = cv2.threshold(src=image,
                                     thresh=127,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)
        elif method == 'otsu':
            _, image = cv2.threshold(src=image,
                                     thresh=0,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive_mean':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)
        elif method == 'adaptive_gaussian':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)

        return image

    def erode(self, image, kernel_size, iterations=1, radius=True):
        """
        Apply erode transformation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be eroded.
        kernel_size : int
            Kernel size for erosion.
        iterations : int
            Number of iterations for erosion.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations.

        Returns
        -------
        ndarray
            Eroded image.
        """

        if radius:
            kernel_size = np.random.randint(1, kernel_size + 1)
            iterations = np.random.randint(1, iterations + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        image = cv2.erode(image, kernel, iterations=iterations)

        return image

    def dilate(self, image, kernel_size, iterations=1, radius=True):
        """
        Apply dilate transformation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be dilated.
        kernel_size : int
            Kernel size for dilation.
        iterations : int
            Number of iterations for dilation.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations.

        Returns
        -------
        ndarray
            Dilated image.
        """

        if radius:
            kernel_size = np.random.randint(1, kernel_size + 1)
            iterations = np.random.randint(1, iterations + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        image = cv2.dilate(image, kernel, iterations=iterations)

        return image

    def elastic(self, image, kernel_size, alpha=1.0, radius=True):
        """
        Apply elastic transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be distorted.
        kernel_size : int
            Kernel size for elastic transform.
        alpha : float
            Factor of elastic transform.
        radius : bool, optional
            Whether to use range radius for kernel size and alpha.

        Returns
        -------
        ndarray
            Distorted image.
        """

        if radius:
            kernel_size = np.random.randint(1, kernel_size + 1)
            alpha = np.random.uniform(0.0, alpha)

        if kernel_size % 2 == 0:
            kernel_size += 1

        dxy = np.random.uniform(-1.0, 1.0, size=image.shape[:2])
        dxy = cv2.GaussianBlur(dxy, (kernel_size, kernel_size), 0) * (kernel_size * alpha)

        org_coords = np.indices((image.shape[0], image.shape[1]), dtype=np.float32).transpose(1, 2, 0)
        displaced_coords = np.float32(org_coords + np.stack((dxy, dxy), axis=-1))

        image = cv2.remap(src=image,
                          map1=displaced_coords[..., 1],
                          map2=displaced_coords[..., 0],
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=np.median(image.flatten()))

        return image

    def perspective(self, image, alpha, radius=True):
        """
        Apply perspective transformation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        alpha : float
            Factor of perspective transformation.
        radius : bool, optional
            Whether to use range radius for type and alpha.

        Returns
        -------
        ndarray
            Transformed image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        height, width = image.shape[:2]

        max_offset = min(height, width)
        max_offset_factor = (1 - (max_offset / (height + width))) ** 4
        max_offset = int(np.ceil(max_offset * alpha * max_offset_factor))

        src_points = np.array([
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ], dtype=np.float32)

        dst_points = np.array([
            (np.random.randint(0, max_offset), np.random.randint(0, max_offset)),
            (np.random.randint(width - max_offset, width), np.random.randint(0, max_offset)),
            (np.random.randint(width - max_offset, width), np.random.randint(height - max_offset, height)),
            (np.random.randint(0, max_offset), np.random.randint(height - max_offset, height)),
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)

        image = cv2.warpPerspective(image, M, (width, height),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=np.median(image.flatten()))

        return image

    def mixup(self, image, batch_images, opacity, iterations=1, radius=True):
        """
        Apply mixup augmentation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be mixed.
        batch_images : list
            List of additional images for mixing.
        opacity : float
            Opacity of the mixup effect.
        iterations : int
            Number of images for the mixup operation.
        radius : bool, optional
            Whether to use range radius for opacity and iterations.

        Returns
        -------
        ndarray
            Mixed image.
        """

        if batch_images is None or len(batch_images) == 0:
            return image

        height, width = image.shape[:2]

        size_min = height * width * 0.75
        size_max = height * width * 1.25
        size_images = [img for img in batch_images if size_min <= img.shape[0] * img.shape[1] <= size_max]

        if size_images:
            iterations = min(iterations, len(size_images))
            indices = np.uint8(np.random.uniform(0, len(size_images), iterations))
            opacities = np.random.uniform(0.0, opacity, iterations) if radius else np.full(iterations, opacity)

            for i, opc in zip(indices, opacities):
                img = size_images[i]

                if self.binarize_params is not None:
                    img = self.binarize(img, *self.binarize_params[1:])

                ratio_width = width / float(img.shape[1])
                ratio_height = height / float(img.shape[0])
                ratio = min(ratio_width, ratio_height)

                dim = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
                img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA)

                delta_w = width - dim[0]
                delta_h = height - dim[1]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)

                img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=np.mean(image))

                image = cv2.addWeighted(image, 1 - opc, img.astype(image.dtype), opc, 0)

        return image

    def shear(self, image, angle, radius=True):
        """
        Apply shear transformation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        angle : float
            shear angle in degrees.
        radius : bool, optional
            Whether to use range radius for angle.

        Returns
        -------
        ndarray
            Sheared image.
        """

        if radius:
            angle = np.random.uniform(-angle, angle)

        height, width = image.shape[:2]

        max_offset_factor = (1 - (height / (height + width))) ** 4
        shear_tan = np.tan(np.radians(angle * max_offset_factor))

        if angle > 0:
            new_width = int(width + height * shear_tan)
            M = np.float32([[1, shear_tan, 0], [0, 1, 0]])

        else:
            new_width = int(width - height * shear_tan)
            M = np.float32([[1, shear_tan, -height * shear_tan], [0, 1, 0]])

        image = cv2.warpAffine(image, M, (new_width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.median(image.flatten()))

        return image

    def scale(self, image, alpha, radius=True):
        """
        Apply scale to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        alpha : float
            scale alpha.
        radius : bool, optional
            Whether to use range radius for alpha.

        Returns
        -------
        ndarray
            Scaled image.
        """

        if radius:
            alpha = np.random.uniform(-alpha, alpha)

        height, width = image.shape[:2]
        ratio = 1 - alpha

        dim = (int(width * ratio), int(height * ratio))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA)

        if alpha > 0:
            padded_image = np.full((height, width), np.median(image.flatten()), dtype=np.uint8)
            padded_image[:image.shape[0], :image.shape[1]] = image

        return image

    def rotate(self, image, angle, radius=True):
        """
        Apply rotate to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be rotated.
        angle : float
            rotate angle in degrees.
        radius : bool, optional
            Whether to use range radius for angle.

        Returns
        -------
        ndarray
            Rotated image.
        """

        if radius:
            angle = np.random.uniform(-angle, angle)

        height, width = image.shape[:2]

        max_offset_factor = (1 - (height / (height + width))) ** 2

        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle * max_offset_factor, 1.0)

        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        image = cv2.warpAffine(image, M, (new_width, new_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.median(image.flatten()))

        return image

    def shift_y(self, image, alpha, radius=True):
        """
        Apply Y-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        alpha : float
            Y-axis translation factor.
        radius : bool, optional
            Whether to use range radius for alphas.

        Returns
        -------
        ndarray
            Translated image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        height, width = image.shape[:2]

        max_offset = min(height, width)
        max_offset_factor = (1 - (max_offset / (height + width))) ** 4

        y_translation = int(np.ceil(max_offset * alpha * max_offset_factor))

        M = np.float32([[1, 0, 0], [0, 1, y_translation]])

        new_height = height + abs(y_translation)

        image = cv2.warpAffine(image, M, (width, new_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.median(image.flatten()))

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        return image

    def shift_x(self, image, alpha, radius=True):
        """
        Apply X-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        alpha : float
            X-axis translation factor.
        radius : bool, optional
            Whether to use range radius for alphas.

        Returns
        -------
        ndarray
            Translated image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        height, width = image.shape[:2]

        max_offset = min(height, width)
        max_offset_factor = (1 - (max_offset / (height + width))) ** 4

        x_translation = int(np.ceil(max_offset * alpha * max_offset_factor))

        M = np.float32([[1, 0, x_translation], [0, 1, 0]])

        new_width = width + abs(x_translation)

        image = cv2.warpAffine(image, M, (new_width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=np.median(image.flatten()))

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        return image

    def salt_and_pepper(self, image, alpha, radius=True):
        """
        Apply salt and pepper noise to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be noised.
        alpha : float
            Percentage of pixels to add noise to.
        radius : bool, optional
            Whether to use range radius for alpha.

        Returns
        -------
        ndarray
            The noisy image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        noise_mask = np.random.uniform(0.0, 1.0, size=image.shape[:2])
        alpha *= 0.25

        image = np.where(noise_mask < alpha, 0, image)
        image = np.where(noise_mask > (1 - alpha), 255, image)

        return image

    def gaussian_noise(self, image, alpha, radius=True):
        """
        Adds Gaussian noise to an image.

        Parameters
        ----------
        image : ndarray
            The input grayscale image.
        alpha : float
            Noise level factor, where larger alpha adds more noise.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations.

        Returns
        -------
        ndarray
            The noisy image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        height, width = image.shape

        mean = np.mean(image)
        std = np.std(image) * alpha

        gauss = np.random.normal(mean, std, (height, width))
        gauss = gauss.reshape(height, width)

        image = np.add(image, gauss)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return image

    def gaussian_blur(self, image, kernel_size, iterations=1, radius=True):
        """
        Apply Gaussian blur to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be blurred.
        kernel_size : int
            Kernel size for Gaussian blur.
        iterations : int
            Number of iterations for Gaussian blur.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations.

        Returns
        -------
        ndarray
            Blurred image.
        """

        if radius:
            kernel_size = np.random.randint(1, kernel_size + 1)
            iterations = np.random.randint(1, iterations + 1)

        if kernel_size % 2 == 0:
            kernel_size += 1

        for _ in range(iterations):
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image
