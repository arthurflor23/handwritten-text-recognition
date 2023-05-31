import cv2
import json
import numpy as np
import concurrent.futures


class Augmentor():

    def __init__(self,
                 elastic_distortion=None,
                 perspective_transform=None,
                 gaussian_noise=None,
                 gaussian_blur=None,
                 shearing=None,
                 scaling=None,
                 rotation=None,
                 translate_x=None,
                 translate_y=None,
                 mixup=None):

        self.elastic_distortion_params = elastic_distortion
        self.perspective_transform_params = perspective_transform
        self.gaussian_noise_params = gaussian_noise
        self.gaussian_blur_params = gaussian_blur
        self.shearing_params = shearing
        self.scaling_params = scaling
        self.rotation_params = rotation
        self.translate_x_params = translate_x
        self.translate_y_params = translate_y
        self.mixup_params = mixup

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Augmentor object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'elastic_distortion': self.elastic_distortion_params,
            'perspective_transform': self.perspective_transform_params,
            'gaussian_noise': self.gaussian_noise_params,
            'gaussian_blur': self.gaussian_blur_params,
            'shearing': self.shearing_params,
            'scaling': self.scaling_params,
            'rotation': self.rotation_params,
            'translate_x': self.translate_x_params,
            'translate_y': self.translate_y_params,
            'mixup': self.mixup_params,
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
            Elastic Distortion      {self.elastic_distortion_params}
            Perspective Transform   {self.perspective_transform_params}

            Gaussian Noise          {self.gaussian_noise_params}
            Gaussian Blur           {self.gaussian_blur_params}

            Shearing                {self.shearing_params}
            Scaling                 {self.scaling_params}

            Rotation                {self.rotation_params}
            Translation X           {self.translate_x_params}
            Translation Y           {self.translate_y_params}

            Mixup                   {self.mixup_params}
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

    def augmentation(self, image, images=None):
        """
        Apply transformations to an image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        images : list
            List of images used for mixup transformation.

        Returns
        -------
        ndarray
            Transformed image.

        """

        # Elastic Distortions
        if self.elastic_distortion_params and np.random.random() < self.elastic_distortion_params[0]:
            image = self.elastic_distortion(image, *self.elastic_distortion_params[1:])

        # Perspective Transforms
        if self.perspective_transform_params and np.random.random() < self.perspective_transform_params[0]:
            image = self.perspective_transform(image, *self.perspective_transform_params[1:])

        # Gaussian Noise
        if self.gaussian_noise_params and np.random.random() < self.gaussian_noise_params[0]:
            image = self.gaussian_noise(image, *self.gaussian_noise_params[1:])

        # Gaussian Blur
        if self.gaussian_blur_params and np.random.random() < self.gaussian_blur_params[0]:
            image = self.gaussian_blur(image, *self.gaussian_blur_params[1:])

        # Shearing
        if self.shearing_params and np.random.random() < self.shearing_params[0]:
            image = self.shearing(image, *self.shearing_params[1:])

        # Scaling
        if self.scaling_params and np.random.random() < self.scaling_params[0]:
            image = self.scaling(image, *self.scaling_params[1:])

        # Rotation
        if self.rotation_params and np.random.random() < self.rotation_params[0]:
            image = self.rotation(image, *self.rotation_params[1:])

        # TranslateX
        if self.translate_x_params and np.random.random() < self.translate_x_params[0]:
            image = self.translate_x(image, *self.translate_x_params[1:])

        # TranslateY
        if self.translate_y_params and np.random.random() < self.translate_y_params[0]:
            image = self.translate_y(image, *self.translate_y_params[1:])

        # Mixup
        if self.mixup_params and np.random.random() < self.mixup_params[0]:
            image = self.mixup(image, images, *self.mixup_params[1:])

        return image

    def elastic_distortion(self, image, grid_size, magnitude, range_radius=True):
        """
        Apply elastic distortion to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be distorted.
        grid_size : int
            Grid size for elastic distortion.
        magnitude : float
            Magnitude of elastic distortion.
        range_radius : bool, optional
            Whether to use range radius for grid size and magnitude, by default True.

        Returns
        -------
        ndarray
            Distorted image.
        """

        # do something
        # ...

        return image

    def perspective_transform(self, image, type, magnitude, range_radius=True):
        """
        Apply perspective transform to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be transformed.
        type : int
            Type of perspective transform.
        magnitude : float
            Magnitude of perspective transform.
        range_radius : bool, optional
            Whether to use range radius for type and magnitude, by default True.

        Returns
        -------
        ndarray
            Transformed image.
        """

        # do something
        # ...

        return image

    def gaussian_noise(self, image, kernel_size, iterations, range_radius=True):
        """
        Apply Gaussian noise to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be noised.
        kernel_size : int
            Kernel size for Gaussian noise.
        iterations : int
            Number of iterations for Gaussian noise.
        range_radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

        Returns
        -------
        ndarray
            Noised image.
        """

        # do something
        # ...

        return image

    def gaussian_blur(self, image, kernel_size, iterations, range_radius=True):
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
        range_radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

        Returns
        -------
        ndarray
            Blurred image.
        """

        # do something
        # ...

        return image

    def shearing(self, image, value, range_radius=True):
        """
        Apply shearing to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        value : float
            Shearing value.
        range_radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Sheared image.
        """

        # do something
        # ...

        return image

    def scaling(self, image, value, range_radius=True):
        """
        Apply scaling to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        value : float
            Scaling value.
        range_radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Scaled image.
        """

        # do something
        # ...

        return image

    def rotation(self, image, value, range_radius=True):
        """
        Apply rotation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be rotated.
        value : float
            Rotation value.
        range_radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Rotated image.
        """

        # do something
        # ...

        return image

    def translate_x(self, image, value, range_radius=True):
        """
        Apply X-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        value : float
            X-axis translation value.
        range_radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        # do something
        # ...

        return image

    def translate_y(self, image, value, range_radius=True):
        """
        Apply Y-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        value : float
            Y-axis translation value.
        range_radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        # do something
        # ...

        return image

    def mixup(self, image, images, opacity, iterations, range_radius=True):
        """
        Apply mixup augmentation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be mixed.
        images : list
            List of additional images for mixing.
        opacity : float
            Opacity of the mixup effect.
        iterations : int
            Number of iterations for the mixup operation.
        range_radius : bool, optional
            Whether to use range radius for opacity and iterations, by default True.

        Returns
        -------
        ndarray
            Mixed image.
        """

        # do something
        # ...

        return image
