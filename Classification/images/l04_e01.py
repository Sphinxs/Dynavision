from skimage import measure

from os import listdir, path

from imageio import imread

from math import pi

import numpy as np

from sklearn import preprocessing

# Load M Peg 7

mpeg7Path = './data/mpeg7'

mpeg7 = {}

for className in listdir(mpeg7Path):
    mpeg7.update(
        {
            className: {
                image: imread(path.join(mpeg7Path, className, image)) for image in sorted(listdir(path.join(mpeg7Path, className)))
            }
        }
    )

# Get morphological properties

mpeg7Properties = {}

mpeg7Data = np.array([0, 0, 0, 0, 0])

mpeg7LabelsCache = []

for className, imagesDict in mpeg7.items():
    mpeg7Properties.update({className: {}})

    for imageName, imageMatrix in imagesDict.items():
        # Get labels

        imageLabeled = measure.label(imageMatrix, background=0)

        # Get properties

        for prop in measure.regionprops(imageLabeled, imageMatrix):
            mpeg7Properties[className].update({
                imageName: [
                    prop.area,
                    prop.eccentricity,
                    prop.mean_intensity,
                    prop.solidity,
                    (4 * pi * prop.area) /
                    prop.perimeter ** float(2)  # Circularity
                ]
            })

            mpeg7Data = np.vstack(
                (mpeg7Data, mpeg7Properties[className][imageName])
            )

            mpeg7LabelsCache.append(className)

lp = preprocessing.LabelEncoder()

lp.fit(mpeg7LabelsCache)

mpeg7Labels = lp.transform(mpeg7LabelsCache)

mpeg7LabelsNames = lp.classes_