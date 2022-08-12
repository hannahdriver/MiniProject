### BMIF804 Mini Project
## By: Hannah Driver (10090525)
## https://github.com/hannahdriver/BMIF804_MiniProject
# This program takes in an MRI prostate image, and segments the prostate. It then compares the segmentation of the
# prostate to a gold standard segmentation, and calculates the Dice Similarity Coefficient. Using the gold standard
# segment, a biopsy target location is identified, and the pixel intensities from a cubic region around the biopsy
# coordinates are analyzed and plotted.

from utils import *


def main():

    #Read in images and cast to Float32 type
    img = readImage("case23_resampled.nii")
    segment = readImage("case23_resampled_segmentation.nii")
    img32 = sitk.Cast(img, sitk.sitkFloat32)

    ###Part A
    #Filter image
    filtered_img = preSegmentationFilter(img32, 11, 250, 445, 3, 12, 8)

    #Set seeds for segmenting
    fid1 = (181, 116, 32)
    fid2 = (109, 195, 32)
    fid3 = (133, 156, 32)
    fid4 = (231, 186, 32)

    #Segment
    segmented_prostate = prostate_segmenter(filtered_img, fid1, fid2, fid3, fid4, 50, 115)

    #View overlayed segments (newly generated segment and gold standard)
    viewSegmentOverlay(img,segmented_prostate,35)
    viewSegmentOverlay(img,segment,35)

    ###Part B
    #Calculate Dice Similarity Coefficient between generated prostate segment and gold standard
    seg_eval_dice(segment,segmented_prostate)

    ###Part C
    #Identify best location for biopsy target
    biopsy_coordinates = get_target_loc(segment,img)

    ###Part D
    #Extract pixel intensities from cubic region around biopsy target
    pixel_extract(img,biopsy_coordinates,6)


main()