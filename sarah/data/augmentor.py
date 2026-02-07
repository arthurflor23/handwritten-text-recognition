import cv2
import numpy as np


class Augmentor():
    """
    Image augmentation class for applying various transformations to images.
    """

    def __init__(self,
                 erode=None,
                 dilate=None,
                 elastic=None,
                 perspective=None,
                 shear=None,
                 scale=None,
                 rotate=None,
                 shift_y=None,
                 shift_x=None,
                 mixup=None,
                 salt_and_pepper=None,
                 gaussian_noise=None,
                 gaussian_blur=None,
                 seed=None):
        """
        Initializes a new instance of the Augmentor class.

        Parameters
        ----------
        erode : list or None, optional
            Parameters for erode transformation.
        dilate : list or None, optional
            Parameters for dilate transformation.
        elastic : list or None, optional
            Parameters for elastic transformation.
        perspective : list or None, optional
            Parameters for perspective transform transformation.
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
        mixup : list or None, optional
            Parameters for mixup transformation.
        salt_and_pepper : list or None, optional
            Parameters for salt and pepper noise.
        gaussian_noise : list or None, optional
            Parameters for gaussian noise.
        gaussian_blur : list or None, optional
            Parameters for Gaussian blur transformation.
        seed : int or None, optional
            Seed for random values from numpy.
        """

        if seed is not None:
            np.random.seed(seed)

        self.erode_params = erode
        self.dilate_params = dilate
        self.elastic_params = elastic
        self.perspective_params = perspective
        self.shear_params = shear
        self.scale_params = scale
        self.rotate_params = rotate
        self.shift_y_params = shift_y
        self.shift_x_params = shift_x
        self.mixup_params = mixup
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

        def _format(x):
            return tuple(round(y, 2) if isinstance(y, (int, float)) else y for y in x) if x else '-'

        pad, width = 25, 68

        info = "=" * width
        info += f"\n{self.__class__.__name__.center(width)}"
        info += "\n" + "-" * width
        info += f"\n{'erode':<{pad}}: {_format(self.erode_params)}"
        info += f"\n{'dilate':<{pad}}: {_format(self.dilate_params)}"
        info += f"\n{'elastic':<{pad}}: {_format(self.elastic_params)}"
        info += f"\n{'perspective':<{pad}}: {_format(self.perspective_params)}"
        info += f"\n{'shear':<{pad}}: {_format(self.shear_params)}"
        info += f"\n{'scale':<{pad}}: {_format(self.scale_params)}"
        info += f"\n{'rotate':<{pad}}: {_format(self.rotate_params)}"
        info += f"\n{'shift_y':<{pad}}: {_format(self.shift_y_params)}"
        info += f"\n{'shift_x':<{pad}}: {_format(self.shift_x_params)}"
        info += f"\n{'mixup':<{pad}}: {_format(self.mixup_params)}"
        info += f"\n{'salt_and_pepper':<{pad}}: {_format(self.salt_and_pepper_params)}"
        info += f"\n{'gaussian_noise':<{pad}}: {_format(self.gaussian_noise_params)}"
        info += f"\n{'gaussian_blur':<{pad}}: {_format(self.gaussian_blur_params)}"
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

        if self.mixup_params is not None:
            self.mixup_params = list(self.mixup_params[:2]) + [batch_images]

        transformations = [
            (self.erode, self.erode_params, 32),
            (self.dilate, self.dilate_params, 32),
            (self.elastic, self.elastic_params, 32),
            (self.perspective, self.perspective_params, 32),
            (self.shear, self.shear_params, 32),
            (self.scale, self.scale_params, 32),
            (self.rotate, self.rotate_params, 32),
            (self.shift_y, self.shift_y_params, 32),
            (self.shift_x, self.shift_x_params, 32),
            (self.mixup, self.mixup_params, 32),
            (self.salt_and_pepper, self.salt_and_pepper_params, 32),
            (self.gaussian_noise, self.gaussian_noise_params, 32),
            (self.gaussian_blur, self.gaussian_blur_params, 32),
        ]

        for func, params, min_length in transformations:
            if params is None or len(params) == 0 or params[0] <= 0:
                continue

            if min(image.shape[:2]) <= min_length:
                continue

            if np.random.random() <= params[0]:
                image = func(image, *params[1:])

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

        ksize = [(1, kernel_size), (kernel_size, 1), (kernel_size, kernel_size)]
        ksize = ksize[np.random.randint(len(ksize))]

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=ksize)
        image = cv2.erode(src=image, kernel=kernel, iterations=iterations)

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

        ksize = [(1, kernel_size), (kernel_size, 1), (kernel_size, kernel_size)]
        ksize = ksize[np.random.randint(len(ksize))]

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=ksize)
        image = cv2.dilate(src=image, kernel=kernel, iterations=iterations)

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
        dxy = cv2.GaussianBlur(dxy, (kernel_size, kernel_size), 0) * kernel_size * alpha

        org_coords = np.indices((image.shape[0], image.shape[1]), dtype=np.float32).transpose(1, 2, 0)
        displaced_coords = np.float32(org_coords + np.stack((dxy, dxy), axis=-1))

        image = cv2.remap(src=image,
                          map1=displaced_coords[..., 1],
                          map2=displaced_coords[..., 0],
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=int(np.median(image)))

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

        max_factor = (1 - (min(height, width) / (height + width))) ** 4
        dxy_factor = int(np.ceil(min(height, width) * alpha * max_factor))

        src_points = np.array([
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ], dtype=np.float32)

        dst_points = np.array([
            (np.random.randint(0, dxy_factor), np.random.randint(0, dxy_factor)),
            (np.random.randint(width - dxy_factor, width), np.random.randint(0, dxy_factor)),
            (np.random.randint(width - dxy_factor, width), np.random.randint(height - dxy_factor, height)),
            (np.random.randint(0, dxy_factor), np.random.randint(height - dxy_factor, height)),
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)

        image = cv2.warpPerspective(src=image,
                                    M=M,
                                    dsize=(width, height),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=int(np.median(image)))

        return image

    def shear(self, image, alpha, radius=True):
        """
        Apply shear transformation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        alpha : float
            Factor of shear transformation.
        radius : bool, optional
            Whether to use range radius for alpha.

        Returns
        -------
        ndarray
            Sheared image.
        """

        if radius:
            alpha = np.random.uniform(-alpha, alpha)

        height, width = image.shape[:2]

        dy_factor = (1 - (height / (height + width))) ** 4
        angle = alpha * dy_factor

        if angle > 0:
            new_width = int(width + height * angle)
            M = np.float32([[1, angle, 0], [0, 1, 0]])

        else:
            new_width = int(width - height * angle)
            M = np.float32([[1, angle, -height * angle], [0, 1, 0]])

        image = cv2.warpAffine(src=image,
                               M=M,
                               dsize=(new_width, height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=int(np.median(image)))

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
        image = cv2.resize(src=image, dsize=dim, interpolation=cv2.INTER_CUBIC)

        if alpha > 0:
            padded_image = np.full((height, width), int(np.median(image)), dtype=np.uint8)
            padded_image[:image.shape[0], :image.shape[1]] = image

        return image

    def rotate(self, image, alpha, radius=True):
        """
        Apply rotate to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be rotated.
        alpha : float
            Factor of rotate transformation.
        radius : bool, optional
            Whether to use range radius for alpha.

        Returns
        -------
        ndarray
            Rotated image.
        """

        if radius:
            alpha = np.random.uniform(-alpha, alpha)

        height, width = image.shape[:2]

        dy_factor = (1 - (height / (height + width))) ** 4
        angle = alpha * dy_factor

        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)

        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        image = cv2.warpAffine(src=image,
                               M=M,
                               dsize=(new_width, new_height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=int(np.median(image)))

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

        dy_factor = (1 - (height / (height + width))) ** 4
        y_translation = int(np.ceil(height * alpha * dy_factor))

        M = np.float32([[1, 0, 0], [0, 1, y_translation]])

        image = cv2.warpAffine(src=image,
                               M=M,
                               dsize=(width, height + abs(y_translation)),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=int(np.median(image)))

        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

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

        dx_factor = (1 - (width / (height + width))) ** 4
        x_translation = int(np.ceil(width * alpha * dx_factor))

        M = np.float32([[1, 0, x_translation], [0, 1, 0]])

        image = cv2.warpAffine(src=image,
                               M=M,
                               dsize=(width + abs(x_translation), height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=int(np.median(image)))

        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        return image

    def mixup(self, image, opacity, batch_images=None, iterations=1, radius=True):
        """
        Apply mixup augmentation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be mixed.
        opacity : float
            Opacity of the mixup effect.
        batch_images : list, optional
            List of additional images for mixing.
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

                ratio_width = width / float(img.shape[1])
                ratio_height = height / float(img.shape[0])
                ratio = min(ratio_width, ratio_height)

                dim = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
                img = cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_CUBIC)

                delta_w = width - dim[0]
                delta_h = height - dim[1]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)

                img = cv2.copyMakeBorder(src=img,
                                         top=top,
                                         bottom=bottom,
                                         left=left,
                                         right=right,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=int(np.median(image)))

                image = cv2.addWeighted(src1=image,
                                        src2=img.astype(image.dtype),
                                        alpha=1 - opc,
                                        beta=opc,
                                        gamma=0)

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

        image = cv2.normalize(src=np.add(image, gauss),
                              dst=None,
                              alpha=0,
                              beta=255,
                              norm_type=cv2.NORM_MINMAX)

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

        ksize = [(1, kernel_size), (kernel_size, 1), (kernel_size, kernel_size)]
        ksize = ksize[np.random.randint(len(ksize))]

        for _ in range(iterations):
            image = cv2.GaussianBlur(src=image, ksize=ksize, sigmaX=0)

        return image
