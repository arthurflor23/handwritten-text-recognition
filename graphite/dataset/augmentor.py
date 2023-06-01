import cv2
import json
import numpy as np


class Augmentor():
    """
    Image augmentation class for applying various transformations to images.
    """

    def __init__(self,
                 erosion=None,
                 dilation=None,
                 elastic_transform=None,
                 mixup=None,
                 perspective_transform=None,
                 salt_and_pepper=None,
                 gaussian_blur=None,
                 shearing=None,
                 scaling=None,
                 rotation=None,
                 translation=None,
                 reference_pixels=None,
                 seed=None):
        """
        Initializes a new instance of the Augmentor class.

        Parameters
        ----------
        erosion : dict or None, optional
            Parameters for erosion transformation, by default None.
        dilation : dict or None, optional
            Parameters for dilation transformation, by default None.
        elastic_transform : dict or None, optional
            Parameters for elastic transformation, by default None.
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
        reference_pixels : list, optional
            Reference pixel values for transformation, by default None.
        seed : int or None, optional
            Seed for random number generation, by default None.

        Returns
        -------
        None
        """

        np.random.seed(seed)

        self.erosion_params = erosion
        self.dilation_params = dilation
        self.elastic_transform_params = elastic_transform
        self.mixup_params = mixup
        self.perspective_transform_params = perspective_transform
        self.salt_and_pepper_params = salt_and_pepper
        self.gaussian_blur_params = gaussian_blur
        self.shearing_params = shearing
        self.scaling_params = scaling
        self.rotation_params = rotation
        self.translation_params = translation
        self.reference_pixels = reference_pixels or [0, 127, 255]
        self.seed = seed

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Augmentor object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'erosion': self.erosion_params,
            'dilation': self.dilation_params,
            'elastic_transform': self.elastic_transform_params,
            'mixup': self.mixup_params,
            'perspective_transform': self.perspective_transform_params,
            'gaussian_blur': self.gaussian_blur_params,
            'reference_pixels': self.reference_pixels,
            'shearing': self.shearing_params,
            'scaling': self.scaling_params,
            'rotation': self.rotation_params,
            'translation': self.translation_params,
            'salt_and_pepper': self.salt_and_pepper_params,
            'seed': self.seed,
        })

    def __str__(self):
        """
        Returns a string representation of the Augmentor object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Augmentor Configuration\n
            Erosion                 {self.erosion_params}
            Dilation                {self.dilation_params}
            Elastic Transform       {self.elastic_transform_params}

            Mixup                   {self.mixup_params}
            Perspective Transform   {self.perspective_transform_params}

            Salt and Pepper Noise   {self.salt_and_pepper_params}
            Gaussian Blur           {self.gaussian_blur_params}

            Shearing                {self.shearing_params}
            Scaling                 {self.scaling_params}
            Rotation                {self.rotation_params}
            Translation             {self.translation_params}

            Reference Pixels        {self.reference_pixels}
            Seed                    {self.seed}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def augmentation(self, image, batch_images):
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

        transformations = [
            (self.erosion, self.erosion_params),
            (self.dilation, self.dilation_params),
            (self.elastic_transform, self.elastic_transform_params),
            (self.mixup, self.mixup_params[:1] + [batch_images] + self.mixup_params[1:]),
            (self.perspective_transform, self.perspective_transform_params),
            (self.salt_and_pepper, self.salt_and_pepper_params),
            (self.gaussian_blur, self.gaussian_blur_params),
            (self.shearing, self.shearing_params),
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

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
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

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        image = cv2.dilate(image, kernel, iterations=iterations)

        return image

    def elastic_transform(self, image, kernel_size, radius=True):
        """
        Apply elastic transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be distorted.
        kernel_size : int
            Kernel size for elastic transform.
        radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

        Returns
        -------
        ndarray
            Distorted image.
        """

        if radius:
            kernel_size = np.random.randint(1, kernel_size + 1)

        if kernel_size % 2 == 0:
            kernel_size += 1

        dxy = self._cv2_randu(image.shape[:2], -1.0, 1.0)
        dxy = cv2.GaussianBlur(dxy, (kernel_size, kernel_size), 0) * (kernel_size * 0.5)

        org_coords = np.indices((image.shape[0], image.shape[1]), dtype=np.float32).transpose(1, 2, 0)
        displaced_coords = np.float32(org_coords + np.stack((dxy, dxy), axis=-1))

        image = cv2.remap(src=image,
                          map1=displaced_coords[..., 1],
                          map2=displaced_coords[..., 0],
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=self.reference_pixels[1])

        return image

    def mixup(self, image, batch_images, opacity, pickups, radius=True):
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
        pickups : int
            Number of images for the mixup operation.
        radius : bool, optional
            Whether to use range radius for opacity and pickups, by default True.

        Returns
        -------
        ndarray
            Mixed image.
        """

        batch_length = len(batch_images)

        pickups = min(pickups, batch_length)
        pickup_idxs = np.uint8(np.random.uniform(0, batch_length, pickups))

        height, width = image.shape[:2]
        opacity_vals = np.random.uniform(0.0, opacity, pickups) if radius else np.full(pickups, opacity)

        for pickup_idx, pickup_opac in zip(pickup_idxs, opacity_vals):
            pickup_img = batch_images[pickup_idx]

            if pickup_img.shape[:2] != image.shape[:2]:
                pickup_img = cv2.resize(pickup_img, (width, height), interpolation=cv2.INTER_LINEAR)

            image = cv2.addWeighted(image, 1 - pickup_opac, pickup_img, pickup_opac, 0)

        return image

    def perspective_transform(self, image, factor, radius=True):
        """
        Apply perspective transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        factor : float
            Factor of perspective transform.
        radius : bool, optional
            Whether to use range radius for type and factor, by default True.

        Returns
        -------
        ndarray
            Transformed image.
        """

        if radius:
            factor = np.random.uniform(0.0, factor)

        height, width = image.shape[:2]
        max_offset = max(1, int(min(height, width) * factor))

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

        image = cv2.warpPerspective(image, M, image.shape[::-1],
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=self.reference_pixels[1])

        return image

    def salt_and_pepper(self, image, noise_percentage, radius=True):
        """
        Apply Gaussian noise to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be noised.
        noise_percentage : float
            Percentage of pixels to add noise to (between 0 and 1).
        radius : bool, optional
            Whether to use range radius for noise_percentage, by default True.

        Returns
        -------
        ndarray
            Noised image.
        """

        if radius:
            noise_percentage = np.random.uniform(0.0, noise_percentage)

        noise_mask = self._cv2_randu(image.shape[:2], 0.0, 1.0)
        noise_percentage *= 0.25

        image = np.where(noise_mask < noise_percentage, self.reference_pixels[0], image)
        image = np.where(noise_mask > (1 - noise_percentage), self.reference_pixels[2], image)

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

    def shearing(self, image, factor, radius=True):
        """
        Apply shearing to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        factor : float
            Shearing factor.
        radius : bool, optional
            Whether to use range radius for factor, by default True.

        Returns
        -------
        ndarray
            Sheared image.
        """

        if radius:
            factor = np.random.uniform(-factor, factor)

        height, width = image.shape[:2]

        extra_width = int(abs(factor) * height)
        extended_image = cv2.copyMakeBorder(image, 0, 0, extra_width, extra_width,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=self.reference_pixels[1])

        M = np.float32([[1, factor, 0], [0, 1, 0]])

        image = cv2.warpAffine(extended_image, M, (width + 2 * extra_width, height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.reference_pixels[1])

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        return image

    def scaling(self, image, factor, radius=True):
        """
        Apply scaling to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        factor : float
            Scaling factor.
        radius : bool, optional
            Whether to use range radius for factor, by default True.

        Returns
        -------
        ndarray
            Scaled image.
        """

        if radius:
            factor = np.random.uniform(-factor, factor)

        height, width = image.shape[:2]

        new_height = int(height * (1 - factor))
        new_width = int(width * (1 - factor))

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

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
            Whether to use range radius for factor, by default True.

        Returns
        -------
        ndarray
            Rotated image.
        """

        if radius:
            angle = np.random.uniform(-angle, angle)

        height, width = image.shape[:2]

        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        image = cv2.warpAffine(image, M, (new_width, new_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.reference_pixels[1])

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        return image

    def translation(self, image, y_factor, x_factor, radius=True):
        """
        Apply X-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        y_factor : float
            Y-axis translation factor.
        x_factor : float
            X-axis translation factor.
        radius : bool, optional
            Whether to use range radius for factor, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        if radius:
            y_factor = np.random.uniform(0.0, y_factor)
            x_factor = np.random.uniform(0.0, x_factor)

        height, width = image.shape[:2]
        max_offset = min(height, width)

        abs_y_translation = int(max_offset * y_factor)
        abs_x_translation = int(max_offset * x_factor)

        M = np.float32([[1, 0, abs_x_translation], [0, 1, abs_y_translation]])

        new_height = height + abs(abs_y_translation)
        new_width = width + abs(abs_x_translation)

        image = cv2.warpAffine(image, M, (new_width, new_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.reference_pixels[1])

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

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
