import os
from PIL import Image


BASE_PATH = f"{os.path.dirname(__file__)}/../"

example = os.path.join(BASE_PATH, "data/iam/sentences/a01/a01-000u/a01-000u-s00-00.png")


print(__file__)
print(example)
print(os.path.basename(__file__))
print(os.path.dirname(__file__))

im = Image.open(example)

im.show()
