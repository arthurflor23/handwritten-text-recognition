import cv2
import json
import numpy as np


class Augmentor():
    """
    Image augmentation class for applying various transformations to images.
    """

    def __init__(self,
                 elastic_transform=None,
                 erosion=None,
                 dilation=None,
                 mixup=None,
                 perspective_transform=None,
                 salt_and_pepper=None,
                 gaussian_blur=None,
                 shearing=None,
                 scaling=None,
                 rotation=None,
                 translation=None,
                 pad_value=255,
                 disable_augmentation=False,
                 seed=None):
        """
        Initializes a new instance of the Augmentor class.

        Parameters
        ----------
        elastic_transform : dict or None, optional
            Parameters for elastic transformation, by default None.
        erosion : dict or None, optional
            Parameters for erosion transformation, by default None.
        dilation : dict or None, optional
            Parameters for dilation transformation, by default None.
        mixup : dict or None, optional
            Parameters for mixup transformation, by default None.
        perspective_transform : dict or None, optional
            Parameters for perspective transform transformation, by default None.
        salt_and_pepper : dict or None, optional
            Parameters for salt and pepper noise, by default None.
        gaussian_blur : dict or None, optional
            Parameters for Gaussian blur transformation, by default None.
        shearing : dict or None, optional
            Parameters for shearing transformation, by default None.
        scaling : dict or None, optional
            Parameters for scaling transformation, by default None.
        rotation : dict or None, optional
            Parameters for rotation transformation, by default None.
        translation : dict or None, optional
            Parameters for vertical and horizontal translation transformation, by default None.
        pad_value : int, optional
            Padding value. Default is 255.
        disable_augmentation : bool,
            Flag to disable augmentation, by default False.
        seed : int or None, optional
            Seed for random number generation, by default None.

        Returns
        -------
        None
        """

        np.random.seed(seed)

        self.elastic_transform_params = elastic_transform
        self.erosion_params = erosion
        self.dilation_params = dilation
        self.mixup_params = mixup
        self.perspective_transform_params = perspective_transform
        self.salt_and_pepper_params = salt_and_pepper
        self.gaussian_blur_params = gaussian_blur
        self.shearing_params = shearing
        self.scaling_params = scaling
        self.rotation_params = rotation
        self.translation_params = translation

        self.pad_value = pad_value
        self.disable_augmentation = disable_augmentation
        self.seed = seed

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'elastic_transform': self.elastic_transform_params,
            'erosion': self.erosion_params,
            'dilation': self.dilation_params,
            'mixup': self.mixup_params,
            'perspective_transform': self.perspective_transform_params,
            'gaussian_blur': self.gaussian_blur_params,
            'shearing': self.shearing_params,
            'scaling': self.scaling_params,
            'rotation': self.rotation_params,
            'translation': self.translation_params,
            'salt_and_pepper': self.salt_and_pepper_params,
            'pad_value': self.pad_value,
            'disable_augmentation': self.disable_augmentation,
            'seed': self.seed,
        }

        attributes = json.dumps(attributes, default=lambda x: str(x))

        return attributes

    def __str__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Augmentor Configuration\n
            Elastic Transform       {self.elastic_transform_params}
            Erosion                 {self.erosion_params}
            Dilation                {self.dilation_params}

            Mixup                   {self.mixup_params}
            Perspective Transform   {self.perspective_transform_params}

            Salt and Pepper Noise   {self.salt_and_pepper_params}
            Gaussian Blur           {self.gaussian_blur_params}

            Shearing                {self.shearing_params}
            Scaling                 {self.scaling_params}
            Rotation                {self.rotation_params}
            Translation             {self.translation_params}

            Padding Value           {self.pad_value}
            Augmentation Disabled   {self.disable_augmentation}
            Seed                    {self.seed}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def augmentation(self, image, batch_images=None):
        """
        Apply transformations to an image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        batch_images : list
            List of images used for mixup transformation, default is None.

        Returns
        -------
        ndarray
            Transformed image.
        """

        if not self.disable_augmentation:
            transformations = [
                # (self.elastic_transform, self.elastic_transform_params),
                (self.erosion, self.erosion_params),
                (self.dilation, self.dilation_params),
                # (self.mixup, self.mixup_params + [batch_images] if self.mixup_params else None),
                # (self.perspective_transform, self.perspective_transform_params),
                # (self.salt_and_pepper, self.salt_and_pepper_params),
                # (self.gaussian_blur, self.gaussian_blur_params),
                # (self.shearing, self.shearing_params),
                (self.scaling, self.scaling_params),
                (self.rotation, self.rotation_params),
                (self.translation, self.translation_params),
            ]

            for transform_func, params in transformations:
                if params and np.random.random() < params[0]:
                    image = transform_func(image, *params[1:])

        return image

    def erosion(self, image, kernel_size, iterations, radius=True):
        """
        Apply Erosion to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be eroded.
        kernel_size : int
            Kernel size for erosion.
        iterations : int
            Number of iterations for erosion.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

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

    def dilation(self, image, kernel_size, iterations, radius=True):
        """
        Apply Dilation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be dilated.
        kernel_size : int
            Kernel size for dilation.
        iterations : int
            Number of iterations for dilation.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

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

    def elastic_transform(self, image, kernel_size, alpha, radius=True):
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
            Whether to use range radius for kernel size and iterations, by default True.

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

        dxy = self._cv2_randu(image.shape[:2], -1.0, 1.0)
        dxy = cv2.GaussianBlur(dxy, (kernel_size, kernel_size), 0) * (kernel_size * alpha)

        org_coords = np.indices((image.shape[0], image.shape[1]), dtype=np.float32).transpose(1, 2, 0)
        displaced_coords = np.float32(org_coords + np.stack((dxy, dxy), axis=-1))

        image = cv2.remap(src=image,
                          map1=displaced_coords[..., 1],
                          map2=displaced_coords[..., 0],
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=self.pad_value)

        return image

    def mixup(self, image, opacity, iterations, batch_images=None, radius=True):
        """
        Apply mixup augmentation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be mixed.
        opacity : float
            Opacity of the mixup effect.
        iterations : int
            Number of images for the mixup operation.
        batch_images : list
            List of additional images for mixing.
        radius : bool, optional
            Whether to use range radius for opacity and iterations, by default True.

        Returns
        -------
        ndarray
            Mixed image.
        """

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
                img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA)

                delta_w = width - dim[0]
                delta_h = height - dim[1]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)

                img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=np.mean(image))

                image = cv2.addWeighted(image, 1 - opc, img, opc, 0)

        return image

    def perspective_transform(self, image, alpha, radius=True):
        """
        Apply perspective transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        alpha : float
            Factor of perspective transform.
        radius : bool, optional
            Whether to use range radius for type and alpha, by default True.

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
                                    borderValue=self.pad_value)

        return image

    def salt_and_pepper(self, image, alpha, radius=True):
        """
        Apply Gaussian noise to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be noised.
        alpha : float
            Percentage of pixels to add noise to (between 0 and 1).
        radius : bool, optional
            Whether to use range radius for noise_percentage, by default True.

        Returns
        -------
        ndarray
            Noised image.
        """

        if radius:
            alpha = np.random.uniform(0.0, alpha)

        noise_mask = self._cv2_randu(image.shape[:2], 0.0, 1.0)
        alpha *= 0.25

        image = np.where(noise_mask < alpha, 0, image)
        image = np.where(noise_mask > (1 - alpha), 255, image)

        return image

    def gaussian_blur(self, image, kernel_size, iterations, radius=True):
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
            Whether to use range radius for kernel size and iterations, by default True.

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

    def shearing(self, image, angle, radius=True):
        """
        Apply shearing to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        angle : float
            Shearing angle in degrees.
        radius : bool, optional
            Whether to use range radius for angle, by default True.

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
                               borderValue=self.pad_value)

        return image

    def scaling(self, image, alpha, radius=True):
        """
        Apply scaling to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        alpha : float
            Scaling alpha.
        radius : bool, optional
            Whether to use range radius for alpha, by default True.

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
            padded_image = np.full((height, width), self.pad_value, dtype=np.uint8)
            padded_image[:image.shape[0], :image.shape[1]] = image

        return image

    def rotation(self, image, angle, radius=True):
        """
        Apply rotation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be rotated.
        angle : float
            Rotation angle in degrees.
        radius : bool, optional
            Whether to use range radius for angle, by default True.

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
                               borderValue=self.pad_value)

        return image

    def translation(self, image, y_alpha, x_alpha, radius=True):
        """
        Apply Y and X-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        y_alpha : float
            Y-axis translation factor.
        x_alpha : float
            X-axis translation factor.
        radius : bool, optional
            Whether to use range radius for alphas, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        if radius:
            y_alpha = np.random.uniform(0.0, y_alpha)
            x_alpha = np.random.uniform(0.0, x_alpha)

        height, width = image.shape[:2]

        max_offset = min(height, width)
        max_offset_factor = (1 - (max_offset / (height + width))) ** 4

        y_translation = int(np.ceil(max_offset * y_alpha * max_offset_factor))
        x_translation = int(np.ceil(max_offset * x_alpha * max_offset_factor))

        M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

        new_height = height + abs(y_translation)
        new_width = width + abs(x_translation)

        image = cv2.warpAffine(image, M, (new_width, new_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.pad_value)

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        return image

    def _cv2_randu(self, shape, low, high):
        """
        Generate an array of random numbers using OpenCV's function.

        Parameters
        ----------
        shape : tuple
            Shape of the output array.
        low : float or int
            Lower bound of the random number range.
        high : float or int
            Upper bound of the random number range.

        Returns
        -------
        ndarray
            Array of random numbers with the specified shape and range.
        """

        try:
            self._cv2_seed += 1
            cv2.setRNGSeed(self._cv2_seed)

        except Exception:
            self._cv2_seed = self.seed or 0
            cv2.setRNGSeed(self._cv2_seed)

        rand = cv2.randu(np.empty(shape), low, high)

        return rand
