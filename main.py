### BMIF804 Mini Project
## By: Hannah Driver (10090525)
## https://github.com/hannahdriver/BMIF804_MiniProject
# This program....

from utils import *


def main():

    ###Part 1

    #Read in images and cast to Float32 type
    img = readImage("case23_resampled.nii")
    segment = readImage("case23_resampled_segmentation.nii")
    img = sitk.Cast(img, sitk.sitkFloat32)
    segment = sitk.Cast(segment, sitk.sitkFloat32)

    ###Part 2

    ###Part 3
    get_target_loc(segment)

    ###Part 4

main()