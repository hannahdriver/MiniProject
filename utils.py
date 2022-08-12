import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.chdir("C:\\Users\\User\\Documents\\Biomedical_Informatics\\BMIF804\\Mini_Project")

def readImage(imageName):
    """
    Read in file to create image object
    :param imageName: name of the file to read in
    :return: image object
    """
    image = sitk.ReadImage(imageName)
    return image


def prostate_segmenter(image):
    pass

def seg_eval_dice(seg1, seg2):

    #Make sure origin and spacing are the same for each segment
    standard_origin = seg1.GetOrigin()
    standard_spacing = seg1.GetSpacing()
    seg2.SetOrigin(standard_origin)
    seg2.SetSpacing(standard_spacing)

    #Calculate dice similarity coefficient between segments
    measures = sitk.LabelOverlapMeasuresImageFilter()
    measures.Execute(seg1, seg2)
    DSC = measures.GetDiceCoefficient()
    print(DSC)


def get_target_loc(image):

    # Determine slice in LP plane with largest area (use # white pixels as proxy for largest area)
    segment_array = sitk.GetArrayFromImage(image)

    list_of_pixel_nums = []

    for slice_z in np.rollaxis(segment_array, 0):
        num_white_pixels = np.count_nonzero(slice_z == 1)
        list_of_pixel_nums.append(num_white_pixels)

    max_index = list_of_pixel_nums.index(max(list_of_pixel_nums))

    #Find centroid of the segment
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(image[:, :, max_index] == 1)
    centroid = label_shape_filter.GetCentroid(1)

    dummy_S_coord = image.TransformIndexToPhysicalPoint((0, 0, max_index))
    coordinates = [centroid[0], centroid[1], dummy_S_coord[2]]

    print("The centroid of the prostate in slice {} of the LP plane is: {}, {}, {}.".format(max_index,
                                                                                            np.round(coordinates[0], 2),
                                                                                            np.round(coordinates[1], 2),
                                                                                            np.round(coordinates[2],
                                                                                                     2)))

    # Plot the biopsy point
    x_coord = (coordinates[0] - image.GetOrigin()[0]) / image.GetSpacing()[0]
    y_coord = (coordinates[1] - image.GetOrigin()[1]) / image.GetSpacing()[1]

    plt.figure(figsize=(20, 20))
    plt.gray()
    plt.imshow(sitk.GetArrayFromImage(image[:, :, max_index]))
    plt.scatter(x_coord, y_coord, c='red', marker='x', s=200)
    plt.axis('off')
    plt.title('Biopsy Point')
    plt.show()


def pixel_extract():
    pass


