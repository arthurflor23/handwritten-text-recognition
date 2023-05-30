import cv2
import json
import numpy as np
import concurrent.futures


class Augmentor():

    def __init__(self,
                 mixup=None,
                 dilation=None,
                 elastic_distortion=None,
                 perspective_transform=None,
                 gaussian_noise=None,
                 gaussian_blur=None,
                 shearing=None,
                 scaling=None,
                 rotation=None,
                 translate_x=None,
                 translate_y=None):

        self.mixup_params = mixup
        self.dilation_params = dilation
        self.elastic_distortion_params = elastic_distortion
        self.perspective_transform_params = perspective_transform
        self.gaussian_noise_params = gaussian_noise
        self.gaussian_blur_params = gaussian_blur
        self.shearing_params = shearing
        self.scaling_params = scaling
        self.rotation_params = rotation
        self.translate_x_params = translate_x
        self.translate_y_params = translate_y

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Augmentor object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'mixup': self.mixup_params,
            'dilation': self.dilation_params,
            'elastic_distortion': self.elastic_distortion_params,
            'perspective_transform': self.perspective_transform_params,
            'gaussian_noise': self.gaussian_noise_params,
            'gaussian_blur': self.gaussian_blur_params,
            'shearing': self.shearing_params,
            'scaling': self.scaling_params,
            'rotation': self.rotation_params,
            'translate_x': self.translate_x_params,
            'translate_y': self.translate_y_params,
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
            Mixup                   {self.mixup_params}
            Dilation                {self.dilation_params}
            Elastic Distortion      {self.elastic_distortion_params}
            Perspective Transform   {self.perspective_transform_params}

            Gaussian Noise          {self.gaussian_noise_params}
            Gaussian Blur           {self.gaussian_blur_params}

            Shearing                {self.shearing_params}
            Scaling                 {self.scaling_params}

            Rotation                {self.rotation_params}
            Translation X           {self.translate_x_params}
            Translation Y           {self.translate_y_params}
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

        # Mixup
        if batch_images and self.mixup_params and np.random.random() < self.mixup_params[0]:
            image = self.mixup(image, batch_images, *self.mixup_params[1:])

        # Dilation
        if self.dilation_params and np.random.random() < self.dilation_params[0]:
            image = self.dilation(image, *self.dilation_params[1:])

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
                interp = cv2.INTER_CUBIC if pickup_img.shape[0] > image.shape[0] \
                    or pickup_img.shape[1] > image.shape[1] else cv2.INTER_AREA

                pickup_img = cv2.resize(pickup_img, image.shape[:2][::-1], interpolation=interp)

            image = cv2.addWeighted(image, 1 - pickup_opac, pickup_img, pickup_opac, 0)

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

        # do something
        # ...

        return image

    def elastic_distortion(self, image, grid_size, magnitude, radius=True):
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
        radius : bool, optional
            Whether to use range radius for grid size and magnitude, by default True.

        Returns
        -------
        ndarray
            Distorted image.
        """

        # do something
        # ...

        return image

    def perspective_transform(self, image, type, magnitude, radius=True):
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
        radius : bool, optional
            Whether to use range radius for type and magnitude, by default True.

        Returns
        -------
        ndarray
            Transformed image.
        """

        # do something
        # ...

        return image

    def gaussian_noise(self, image, kernel_size, iterations, radius=True):
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
        radius : bool, optional
            Whether to use range radius for kernel size and iterations, by default True.

        Returns
        -------
        ndarray
            Noised image.
        """

        # do something
        # ...

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

        # do something
        # ...

        return image

    def shearing(self, image, value, radius=True):
        """
        Apply shearing to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be sheared.
        value : float
            Shearing value.
        radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Sheared image.
        """

        # do something
        # ...

        return image

    def scaling(self, image, value, radius=True):
        """
        Apply scaling to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be scaled.
        value : float
            Scaling value.
        radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Scaled image.
        """

        # do something
        # ...

        return image

    def rotation(self, image, value, radius=True):
        """
        Apply rotation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be rotated.
        value : float
            Rotation value.
        radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Rotated image.
        """

        # do something
        # ...

        return image

    def translate_x(self, image, value, radius=True):
        """
        Apply X-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        value : float
            X-axis translation value.
        radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        # do something
        # ...

        return image

    def translate_y(self, image, value, radius=True):
        """
        Apply Y-axis translation to the image.

        Parameters
        ----------
        image : ndarray
            Input image to be translated.
        value : float
            Y-axis translation value.
        radius : bool, optional
            Whether to use range radius for value, by default True.

        Returns
        -------
        ndarray
            Translated image.
        """

        # do something
        # ...

        return image
