import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\User\\Documents\\Biomedical_Informatics\\BMIF804\\Mini_Project")

def readImage(imageName):
    """
    Read in file to create image object
    :param imageName: name of the file to read in
    :return: image object
    """

    image = sitk.ReadImage(imageName)
    return image


def preSegmentationFilter(image, medianKernel, lowThresh, highThresh, cannyVar, cannyLow, cannyHigh):
    """
    Filter a prostate image with median filtering, thresholing, and canny edge detection to prepare for segmentation
    :param image: image object
    :param medianKernel: kernel value for median filtering
    :param lowThresh: lower pixel intensity value for thresholding
    :param highThresh: upper pixel intensity value for thresholding
    :param cannyVar: variance value for canny edge detection
    :param cannyLow: lower threshold value for canny edge detection
    :param cannyHigh: upper threshold value for canny edge detection
    :return: a filtered image object
    """

    #employ median filter
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(medianKernel)
    img_denoised = median_filter.Execute(image)

    #threshold intensities to get rid of unwanted ones
    size = img_denoised.GetSize()
    img_denoised_array = sitk.GetArrayFromImage(img_denoised)
    img_low_thresh_array = np.zeros((size[2], size[1], size[0]))
    img_low_thresh_array = np.where(img_denoised_array < lowThresh, img_low_thresh_array, img_denoised_array)
    img_high_thresh_array = np.zeros((size[2], size[1], size[0]))
    img_thresh_array = np.where(img_low_thresh_array > highThresh, img_high_thresh_array, img_low_thresh_array)
    img_thresh = sitk.GetImageFromArray(img_thresh_array)

    #canny filtering
    canny_filter = sitk.CannyEdgeDetectionImageFilter()
    canny_filter.SetVariance(cannyVar)
    canny_filter.SetLowerThreshold(cannyLow)
    canny_filter.SetUpperThreshold(cannyHigh)
    img_canny = canny_filter.Execute(img_thresh)

    #add canny outlines to thresholded image
    canny_array = sitk.GetArrayFromImage(img_canny)
    new_array = np.where(canny_array == 1, canny_array, img_thresh_array)
    new_image = sitk.GetImageFromArray(new_array)
    new_image_cast = sitk.Cast(new_image, sitk.sitkUInt8)

    return(new_image_cast)


def prostate_segmenter(image,fiducial1,fiducial2,fiducial3,fiducial4,lowerBound,upperBound):
    """
    Segmentation algorithm using SimpleITK's ConnectedThreshold function
    :param image: image object
    :param fiducial1: first seed point
    :param fiducial2: second seed point
    :param fiducial3: third seed point
    :param fiducial4: fourth seed point
    :param lowerBound: lower pixel intensity value for thresholding
    :param upperBound: upper pixel intensity value for thresholding
    :return: an image object of the segmentation
    """

    #segment prostate
    SeedList = [fiducial1,fiducial2,fiducial3,fiducial4]
    segmented = sitk.ConnectedThreshold(image, seedList=SeedList, lower=lowerBound, upper=upperBound)

    #write out segment as an nrrd image
    sitk.WriteImage(segmented, "my_segmentation.nrrd")

    return segmented


def viewSegmentOverlay(image,segment,slice_no):
    """
    Plot a segmentation mask over an image
    :param image: image object
    :param segment: segmentation object
    :param slice_no: slice number in the LP plane in which to visualize the segmentation
    :return: plot of segment overlayed on image
    """

    segment = sitk.Cast(segment, sitk.sitkUInt8)
    image = sitk.Cast(image, sitk.sitkUInt8)

    #Make sure metadata is the same between the image and segment
    standard_spacing = image.GetSpacing()
    standard_origin = image.GetOrigin()
    segment.SetOrigin(standard_origin)
    segment.SetSpacing(standard_spacing)

    #employ overlay
    img_overlay = sitk.LabelOverlay(image, segment)

    #View overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(sitk.GetArrayFromImage(img_overlay[:, :, slice_no]))
    plt.axis('off')


def seg_eval_dice(seg1, seg2):
    """
    Calculate Dice Similarity Coefficient between two segments
    :param seg1: first segment object
    :param seg2: second segment object
    :return: the Dice Similarity Coefficient score
    """

    #Make sure two segments have the same pixel type --> UInt8
    seg1 = sitk.Cast(seg1, sitk.sitkUInt8)

    #Make sure origin and spacing are the same for each segment
    standard_origin = seg1.GetOrigin()
    standard_spacing = seg1.GetSpacing()
    seg2.SetOrigin(standard_origin)
    seg2.SetSpacing(standard_spacing)

    #Calculate dice similarity coefficient between segments
    measures = sitk.LabelOverlapMeasuresImageFilter()
    measures.Execute(seg1, seg2)
    DSC = measures.GetDiceCoefficient()
    print("The Dice Similarity Coefficient for the two segments is: {}.".format(DSC))


def get_target_loc(segment,image):
    """
    Identify target coordinates for a prostate biopsy location
    :param segment: segmented prostate image object
    :param image: image object
    :return: LPS coordinates of the target biopsy location
    """

    # Determine slice in LP plane with largest area (use # white pixels as proxy for largest area)
    segment_array = sitk.GetArrayFromImage(segment)

    list_of_pixel_nums = []

    for slice_z in np.rollaxis(segment_array, 0):
        num_white_pixels = np.count_nonzero(slice_z == 1)
        list_of_pixel_nums.append(num_white_pixels)

    max_index = list_of_pixel_nums.index(max(list_of_pixel_nums))

    #Find centroid of the segment
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segment[:, :, max_index] == 1)
    centroid = label_shape_filter.GetCentroid(1)

    dummy_S_coord = segment.TransformIndexToPhysicalPoint((0, 0, max_index))
    coordinates = [centroid[0], centroid[1], dummy_S_coord[2]]

    print("The biopsy coordinates (at the centroid of the prostate) in slice {} of the LP plane are: {}, {}, {}.".format(max_index,
                                                                                            np.round(coordinates[0], 2),
                                                                                            np.round(coordinates[1], 2),
                                                                                            np.round(coordinates[2],
                                                                                                     2)))

    # Plot the biopsy point
    x_coord = (coordinates[0] - segment.GetOrigin()[0]) / segment.GetSpacing()[0]
    y_coord = (coordinates[1] - segment.GetOrigin()[1]) / segment.GetSpacing()[1]

    plt.figure(figsize=(20, 20))
    plt.gray()
    plt.imshow(sitk.GetArrayFromImage(image[:, :, max_index]))
    plt.scatter(x_coord, y_coord, c='red', marker='x', s=400)
    plt.axis('off')
    plt.title('Biopsy Point', fontsize=40)
    plt.show()

    return(coordinates)


def pixel_extract(img,point,width):
    """
    Boxplot of pixel identities from a cubic region around a designated biopsy location in an image
    :param img: image object
    :param point: tuple of LPS coordinates for biopsy location
    :param width: width of cubic region (in mm) from which to extract pixel intensities
    :return: boxplot displaying pixel intensities from cubic region around designated point
    """

    #Get biopsy coordinates in physical space
    x_coord = (point[0] - img.GetOrigin()[0]) / img.GetSpacing()[0]
    y_coord = (point[1] - img.GetOrigin()[1]) / img.GetSpacing()[1]
    z_coord = (point[2] - img.GetOrigin()[2]) / img.GetSpacing()[2]

    #Find start and end coordinates for each dimension of the cube
    start_0 = round(x_coord - (0.5 * width))
    end_0 = round(x_coord + (0.5 * width))
    start_1 = round(y_coord - (0.5 * width))
    end_1 = round(y_coord + (0.5 * width))
    start_2 = round(z_coord - (0.5 * width))
    end_2 = round(z_coord + (0.5 * width))

    #Pull pixel intensities from cubic segment
    pixel_intensities = img[start_0:end_0, start_1:end_1, start_2:end_2]

    #Plot pixel intensities from cubic segment
    c = "green"
    xlab_LPS = str(round(point[0])) + "," + str(round(point[1])) + "," + str(round(point[2]))
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