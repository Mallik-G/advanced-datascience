from skimage import data
from skimage import io
from skimage import color

coffee = data.coffee()
io.imshow(coffee)

coffee_gray = color.rgb2gray(coffee)
io.imshow(coffee_gray)

