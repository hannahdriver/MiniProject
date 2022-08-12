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
    plt.scatter(x_coord, y_coord, c='red', marker='x', s=300)
    plt.axis('off')
    plt.title('Biopsy Point', fontsize=40)
    plt.show()

    return(coordinates)


def pixel_extract(img,point,width):

    #Get biopsy coordinates in physical space
    x_coord = (point[0] - img.GetOrigin()[0]) / img.GetSpacing()[0]
    y_coord = (point[1] - img.GetOrigin()[1]) / img.GetSpacing()[1]
    z_coord = (point[2] - img.GetOrigin()[2]) / img.GetSpacing()[2]

    #Find start and end coordinates for each dimension of the cube
    start_0 = int(x_coord - (0.5 * width))
    end_0 = int(x_coord + (0.5 * width))
    start_1 = int(y_coord - (0.5 * width))
    end_1 = int(y_coord + (0.5 * width))
    start_2 = int(z_coord - (0.5 * width))
    end_2 = int(z_coord + (0.5 * width))

    #Pull pixel intensities from cubic segment
    pixel_intensities = img[start_0:end_0, start_1:end_1, start_2:end_2]

    #Plot pixel intensities from cubic segment
    c = "green"
    xlab_LPS = str(int(point[0])) + "," + str(int(point[1])) + "," + str(int(point[2]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.boxplot(pixel_intensities, notch=False, patch_artist=True, boxprops=dict(facecolor="palegreen", color=c),
                whiskerprops=dict(color=c), flierprops=dict(color=c, markeredgecolor=c), capprops=dict(color=c),
                medianprops=dict(color=c))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = xlab_LPS
    ax.set_xticklabels(labels)
    plt.title('Pixel Intensities Around Biopsy Point', fontsize=16)
    plt.xlabel("Biopsy Coordinates (LPS)", fontsize=14)
    plt.ylabel("Pixel Intensity", fontsize=14)
    plt.show()