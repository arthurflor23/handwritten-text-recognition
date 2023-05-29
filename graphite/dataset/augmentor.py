import json


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

        self.elastic_distortion = elastic_distortion
        self.perspective_transform = perspective_transform
        self.gaussian_noise = gaussian_noise
        self.gaussian_blur = gaussian_blur
        self.shearing = shearing
        self.scaling = scaling
        self.rotation = rotation
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.mixup = mixup

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Augmentor object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'elastic_distortion': self.elastic_distortion,
            'perspective_transform': self.perspective_transform,
            'gaussian_noise': self.gaussian_noise,
            'gaussian_blur': self.gaussian_blur,
            'shearing': self.shearing,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'translate_x': self.translate_x,
            'translate_y': self.translate_y,
            'mixup': self.mixup
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
            Elastic Distortion      {self.elastic_distortion}
            Perspective Transform   {self.perspective_transform}

            Gaussian Noise          {self.gaussian_noise}
            Gaussian Blur           {self.gaussian_blur}

            Shearing                {self.shearing}
            Scaling                 {self.scaling}

            Rotation                {self.rotation}
            Translation X           {self.translate_x}
            Translation Y           {self.translate_y}

            Mixup                   {self.mixup}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def transform(self, images):

        return images
