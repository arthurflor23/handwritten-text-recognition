import cv2
import json
import numpy as np
import concurrent.futures

from PIL import Image


class Augmentor():
    """
    Image augmentation class for applying various transformations to images.
    """

    def __init__(self,
                 erosion=None,
                 dilation=None,
                 elastic_transform=None,
                 perspective_transform=None,
                 mixup=None,
                 shearing=None,
                 scaling=None,
                 rotation=None,
                 translation=None,
                 salt_and_pepper=None,
                 gaussian_blur=None,
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
        perspective_transform : dict or None, optional
            Parameters for perspective transform transformation, by default None.
        mixup : dict or None, optional
            Parameters for mixup transformation, by default None.
        shearing : dict or None, optional
            Parameters for shearing transformation, by default None.
        scaling : dict or None, optional
            Parameters for scaling transformation, by default None.
        rotation : dict or None, optional
            Parameters for rotation transformation, by default None.
        translation : dict or None, optional
            Parameters for vertical and horizontal translation transformation, by default None.
        salt_and_pepper : dict or None, optional
            Parameters for salt and pepper noise, by default None.
        gaussian_blur : dict or None, optional
            Parameters for Gaussian blur transformation, by default None.
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
        self.perspective_transform_params = perspective_transform
        self.mixup_params = mixup
        self.shearing_params = shearing
        self.scaling_params = scaling
        self.rotation_params = rotation
        self.translation_params = translation
        self.salt_and_pepper_params = salt_and_pepper
        self.gaussian_blur_params = gaussian_blur
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
            'perspective_transform': self.perspective_transform_params,
            'mixup': self.mixup_params,
            'shearing': self.shearing_params,
            'scaling': self.scaling_params,
            'rotation': self.rotation_params,
            'translation': self.translation_params,
            'salt_and_pepper': self.salt_and_pepper_params,
            'gaussian_blur': self.gaussian_blur_params,
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
            Elastic Transform      {self.elastic_transform_params}
            Perspective Transform   {self.perspective_transform_params}
            Mixup                   {self.mixup_params}

            Shearing                {self.shearing_params}
            Scaling                 {self.scaling_params}
            Rotation                {self.rotation_params}
            Translation             {self.translation_params}

            Salt and Pepper Noise   {self.salt_and_pepper_params}
            Gaussian Blur           {self.gaussian_blur_params}

            Seed                    {self.seed}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def batch_augmentation(self, images):
        """
        Apply transformations to a batch of images.

        Parameters
        ----------
        images : list
            List of input images.

        Returns
        -------
        list
            List of augmented images after applying transformations.

        """

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Submit tasks for reading images
        #     futures = [executor.submit(self.augmentation, image, images) for image in images]

        #     # Process the completed tasks
        #     for i, future in enumerate(futures):
        #         images[i] = future.result()

        for i in range(len(images)):
            images[i] = self.augmentation(images[i], images)

        return images

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

        # Erosion
        if self.erosion_params and np.random.random() < self.erosion_params[0]:
            image = self.erosion(image, *self.erosion_params[1:])

        # Dilation
        if self.dilation_params and np.random.random() < self.dilation_params[0]:
            image = self.dilation(image, *self.dilation_params[1:])

        # Elastic Transform
        if self.elastic_transform_params and np.random.random() < self.elastic_transform_params[0]:
            image = self.elastic_transform(image, *self.elastic_transform_params[1:])

        # Perspective Transform
        if self.perspective_transform_params and np.random.random() < self.perspective_transform_params[0]:
            image = self.perspective_transform(image, *self.perspective_transform_params[1:])

        # Mixup
        if batch_images and self.mixup_params and np.random.random() < self.mixup_params[0]:
            image = self.mixup(image, batch_images, *self.mixup_params[1:])

        # Shearing
        if self.shearing_params and np.random.random() < self.shearing_params[0]:
            image = self.shearing(image, *self.shearing_params[1:])

        # Scaling
        if self.scaling_params and np.random.random() < self.scaling_params[0]:
            image = self.scaling(image, *self.scaling_params[1:])

        # Rotation
        if self.rotation_params and np.random.random() < self.rotation_params[0]:
            image = self.rotation(image, *self.rotation_params[1:])

        # Translation
        if self.translation_params and np.random.random() < self.translation_params[0]:
            image = self.translation(image, *self.translation_params[1:])

        # Salt and Pepper Noise
        if self.salt_and_pepper_params and np.random.random() < self.salt_and_pepper_params[0]:
            image = self.salt_and_pepper(image, *self.salt_and_pepper_params[1:])

        # Gaussian Blur
        if self.gaussian_blur_params and np.random.random() < self.gaussian_blur_params[0]:
            image = self.gaussian_blur(image, *self.gaussian_blur_params[1:])

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
            kernel_size = np.random.randint(1, max(1, kernel_size) + 1)
            iterations = np.random.randint(1, max(1, iterations) + 1)

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
            kernel_size = np.random.randint(1, max(1, kernel_size) + 1)
            iterations = np.random.randint(1, max(1, iterations) + 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        image = cv2.dilate(image, kernel, iterations=iterations)

        return image

    def elastic_transform(self, image, grid_size, factor, radius=True):
        """
        Apply elastic transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be distorted.
        grid_size : int
            Grid size for elastic transform.
        factor : float
            Factor of elastic transform.
        radius : bool, optional
            Whether to use range radius for grid size and factor, by default True.

        Returns
        -------
        ndarray
            Distorted image.
        """

        if radius:
            grid_size = np.random.randint(1, max(1, grid_size) + 1)
            factor = np.random.randint(1, max(1, factor) + 1)

        height, width = image.shape[:2]

        horizontal_tiles = grid_size
        vertical_tiles = grid_size

        width_of_square = width // horizontal_tiles
        height_of_square = height // vertical_tiles

        width_of_last_square = width - width_of_square * (horizontal_tiles - 1)
        height_of_last_square = height - height_of_square * (vertical_tiles - 1)

        dimensions = []
        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                x1 = horizontal_tile * width_of_square
                y1 = vertical_tile * height_of_square

                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([x1, y1, x1 + width_of_last_square, y1 + height_of_last_square])

                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([x1, y1, x1 + width_of_square, y1 + height_of_last_square])

                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([x1, y1, x1 + width_of_last_square, y1 + height_of_square])

                else:
                    dimensions.append([x1, y1, x1 + width_of_square, y1 + height_of_square])

        last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = [[x1, y1, x1, y2, x2, y2, x2, y1] for x1, y1, x2, y2 in dimensions]

        polygon_indices = [[i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles]
                           for i in range((vertical_tiles * horizontal_tiles) - 1)
                           if i not in last_row and i not in last_column]

        for a, b, c, d in polygon_indices:
            dx = np.random.randint(-factor, factor + 1)
            dy = np.random.randint(-factor, factor + 1)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = [[dimensions[i], polygons[i]] for i in range(len(dimensions))]
        background_color = int(np.bincount(np.ravel(image)).argmax())

        image = Image.fromarray(image).transform(image.shape[::-1], Image.MESH,
                                                 data=generated_mesh,
                                                 resample=Image.BICUBIC,
                                                 fillcolor=background_color)

        return np.array(image)

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
            factor = np.random.uniform(0.0, max(0.0, factor))

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

        background_color = int(np.bincount(np.ravel(image)).argmax())
        image = cv2.warpPerspective(image, M, image.shape[::-1], borderValue=background_color)

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

        num_imgs = len(batch_images)
        pickups = min(pickups, num_imgs)

        pickup_idxs = np.random.choice(num_imgs, size=pickups, replace=False)
        opacity_vals = np.random.uniform(0.0, opacity, pickups) if radius else np.full(pickups, opacity)

        for pickup_idx, pickup_opac in zip(pickup_idxs, opacity_vals):
            pickup_img = batch_images[pickup_idx]

            if pickup_img.shape[:2] != image.shape[:2]:
                interpolation = cv2.INTER_CUBIC if pickup_img.shape[0] > image.shape[0] \
                    or pickup_img.shape[1] > image.shape[1] else cv2.INTER_AREA

                pickup_img = cv2.resize(pickup_img, image.shape[:2][::-1], interpolation=interpolation)

            image = cv2.addWeighted(image, 1 - pickup_opac, pickup_img, pickup_opac, 0)

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
        background_color = int(np.bincount(np.ravel(image)).argmax())

        extra_width = int(abs(factor) * height)
        extended_image = cv2.copyMakeBorder(image, 0, 0, extra_width, extra_width,
                                            cv2.BORDER_CONSTANT, value=background_color)

        M = np.float32([[1, factor, 0], [0, 1, 0]])
        image = cv2.warpAffine(extended_image, M, (width + 2 * extra_width, height), borderValue=background_color)

        return image

    def scaling(self, image, min_factor, max_factor):
        """
        Apply scaling to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        min_factor : float
            Scaling min factor.
        max_factor : float
            Scaling max factor.

        Returns
        -------
        ndarray
            Scaled image.
        """

        factor = np.random.uniform(min_factor, max_factor)
        height, width = image.shape[:2]

        new_height = int(height * factor)
        new_width = int(width * factor)

        interpolation = cv2.INTER_CUBIC if new_height > height or new_width > width else cv2.INTER_AREA
        image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

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
        background_color = int(np.bincount(np.ravel(image)).argmax())

        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]

        image = cv2.warpAffine(image, M, (new_width, new_height), borderValue=background_color)

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
            y_factor = np.random.uniform(0.0, max(0.0, y_factor))
            x_factor = np.random.uniform(0.0, max(0.0, x_factor))

        height, width = image.shape[:2]
        background_color = int(np.bincount(np.ravel(image)).argmax())

        abs_y_translation = int(min(height, width) * y_factor)
        abs_x_translation = int(min(height, width) * x_factor)

        M = np.float32([[1, 0, abs_x_translation], [0, 1, abs_y_translation]])

        new_height = height + abs(abs_y_translation)
        new_width = width + abs(abs_x_translation)

        image = cv2.warpAffine(image, M, (new_width, new_height), borderValue=background_color)

        return image

    def salt_and_pepper(self, image, percentage_noise, radius=True):
        """
        Apply Gaussian noise to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be noised.
        percentage_noise : float
            Percentage of pixels to add noise to (between 0 and 1).
        radius : bool, optional
            Whether to use range radius for percentage_noise, by default True.

        Returns
        -------
        ndarray
            Noised image.
        """

        if radius:
            percentage_noise = np.random.uniform(1e-8, max(1e-8, percentage_noise))

        num_pixels = image.size
        num_pixels_to_noise = int(num_pixels * percentage_noise)

        indices = np.random.choice(num_pixels, num_pixels_to_noise, replace=False)

        salt_mask = np.random.rand(num_pixels_to_noise) < 0.5
        pepper_mask = ~salt_mask

        noisy_image = image.flatten()

        max_pixel = int(np.bincount(np.ravel(noisy_image[noisy_image > 127])).argmax())
        min_pixel = int(np.bincount(np.ravel(noisy_image[noisy_image < 127])).argmax())

        noisy_image[indices[salt_mask]] = max_pixel
        noisy_image[indices[pepper_mask]] = min_pixel

        image = noisy_image.reshape(image.shape)

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
            kernel_size = np.random.choice(range(1, max(1, kernel_size) + 1, 2))
            iterations = np.random.randint(1, max(1, iterations) + 1)

        for _ in range(iterations):
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image
