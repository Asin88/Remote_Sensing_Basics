# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:47:53 2021

@author: AAKRITI

Basics of Remote Sensing Image Analysis
"""
# == Import modules ==
import os
import imageio
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage import exposure
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import xml.etree.ElementTree as ET
import statsmodels.api as sm
import seaborn as sn
import math

#Read image
#image = imageio.imread("E:/Data Science Projects/Remote Sensing/L3_SAT_8B_V1_86.25E23.75N_F45C05_20Oct09/l3f45c0520oct09_Image1/L3-NF45C05-106-055-20oct09-BAND5.tif")
# == Define functions ==

#Function to get file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)

#Function to plot histogram
def f_plotHistogram(hfig,hax,image, band_number, image_number):
    """
    Function to plot histogram of DNs values of given image in a given band.
    
    Arguments:
        hfig: matplotlib figure
        hax: matplotlib axes
        image:  numpy array of image
        band_number: spectral band number
        image_numer: position of image in sequence of images to be processed

    Returns: 
        hfig: matplolib figure
        hax: matplotlib axes
    """
    i = band_number - 2
    hax[i].hist(image)
    hax[i].set_title(f"Band {band_number}")
#    hax[i].set_xlabel('DN')
#    hax[i].set_ylabel('No of pixels')
    
    return(hfig,hax)
    
#Function to process image
def f_processImage(image, band_number, image_number, hfig, hax, cmap):
    """
    Function to process a given image in a given band. Improves contrast of image 
    using linear stretching.
    
    Arguments:
    image:  numpy array of image
    band_number: spectral band number
    image_numer: position of image in sequence of images to be processed
    hfig: matplotlib figure
    hax: matplotlib axes
    cmap: colour map 
    
    Returns:
    image_contrast_rescale: numpy array of processed image
    hfig: matplotlib figure
    hax: matplotlib axes
    """    
    #View image
#    plt.imshow(image)
    
    print('Type of the image : ' , type(image)) 
    print('Shape of the image : {}'.format(image.shape)) 
    print('Image Height : {}'.format(image.shape[0])) 
    print('Image Width : {}'.format(image.shape[1])) 
    print('Dimension of Image : {}'.format(image.ndim))
    
    print('Image size : {}'.format(image.size)) 
    print('Maximum RGB value in this image : {}'.format(image.max())) 
    print('Minimum RGB value in this image : {}'.format(image.min()))
    
    #Print value of single pixel
    print('Value of single pixel at 100,50 {}'.format(image[100, 50])) 
    
#    #Histogram of DNs
#    hfig,hax = f_plotHistogram(hfig, hax, image, band_number, image_number)
        
    #Check if image is low contrast
#    print("Does image have low contrast? ", exposure.is_low_contrast(image))
    
    #Contrast Stretching
    image_contrast_rescale = exposure.rescale_intensity(image, in_range='image')
    plt.imsave(f_getFilePath(f"reports/Rescale/Image{image_number}_Band{band_number}.tiff"), image_contrast_rescale, cmap=cmap)
    #View image
    #plt.figure(figsize='12,12')
#    plt.imshow(image_contrast_rescale)
    
#    #Histogram of DNs
#    hfig,hax = f_plotHistogram(hfig, hax, image_contrast_rescale, band_number, image_number)
    
#    #Contrast Stretching using histrogram equalization
    image_contrast_rescaled_equalize = exposure.rescale_intensity(image_contrast_rescale, in_range='image')
    plt.imsave(f_getFilePath(f"reports/Equalize/Rescale_Image{image_number}_Band{band_number}.tiff"), image_contrast_rescaled_equalize, cmap=cmap)
    #View image
    #plt.figure(figsize='12,12')
#    plt.imshow(image_contrast_equalize)
#    
#    #Histogram of DNs
#    hfig,hax = f_plotHistogram(hfig,hax,image_contrast_rescaled_equalize, band_number, image_number)

    
    #Constrast improvement using histogram minimum method (haze correction)
    threshold_value = np.amin(image)
    print(f"Threshold value of Image {image_number} in Band {band_number} : {threshold_value}")
    image_haze_corrected = image[:,:] - threshold_value
    image_darkareas = image_haze_corrected[:,:] < 10
    image_haze_corrected[image_darkareas] = 0
    plt.imsave(f_getFilePath(f"reports/HazeCorrection/Image{image_number}_Band{band_number}.tiff"), image_haze_corrected, cmap=cmap)
    
    
    image_darkareas = image_contrast_rescale[:,:] < 25 
    image_contrast_rescale[image_darkareas] = 0
#    bright_areas =  bright_areas1[:,:] < 100
    plt.imsave(f_getFilePath(f"reports/BrightAreas/Image{image_number}_Band{band_number}.tiff"), image_contrast_rescale, cmap=cmap)
#    plt.imshow(image_contrast_rescale)
    
    return(image_contrast_rescaled_equalize, hfig, hax)
    
#Function to get colourmap for images
def f_chooseColourMap(band_number):
    """
    Function to select colour map for image according to band number
    
    Arguements:
        band_number: Band Number
    Returns:
        cmap: selected colourmap
    """
        #Choose colourmap position according to band
    if band_number == 1:
        cmap = "Blues"
    elif band_number == 2:
        cmap = "Greens"
    elif band_number == 3:
        cmap = "Oranges"
    elif band_number == 4:
        cmap = "Reds"
    elif band_number == 5:
        cmap = "YlOrRd"
    else:
        print("Check Band Number")
        exit()
    return(cmap)
    
#Function to get Dhanbad point coordinates
def f_getPointCoordinates(root, x_location, y_location):
    """
    Function to get point coordinates of given location wrt image coordinates
    
    Arguements: 
        root : Root of xml tree
        x_location : longitude of location
        y_location : latitude of location
    Returns:
        x_coord : X-coordinate on image
        y_coord : Y-coordinate on image
    """
    for coverage in root.findall('Coverage'):
        for cov_item in coverage:
            if cov_item.tag == "Upper_left":
                x_min = float(cov_item.text.split()[2][:-2])
                y_max = float(cov_item.text.split()[5][:-1])
            if cov_item.tag == "Lower_right":
                x_max = float(cov_item.text.split()[2][:-2])
                y_min = float(cov_item.text.split()[5][:-1])
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_coord = (1153/x_range)*(x_location-x_min)
    y_coord = (1153/y_range)*(y_max - y_location)
    
    return(x_coord, y_coord)
    
#Function to mark locations on image
def f_markLocations(image):
    """
    Function to mark landmark locations on image
    
    Arguements: 
        image : Image in PIL format
    Returns: 
        image : Marked image in PIL format
    """
    x_ism, y_ism = f_getPointCoordinates(root, 86.4412, 23.8143)
    x_surya, y_surya = f_getPointCoordinates(root, 86.4804, 23.7955)
    x_stadium, y_stadium = f_getPointCoordinates(root, 86.4254, 23.8310)
    x_airport,y_airport = f_getPointCoordinates(root,86.4253,23.8340)
    x_birsa,y_birsa = f_getPointCoordinates(root,86.4155,23.8306)
    x_mohlidih,y_mohlidih = f_getPointCoordinates(root,86.3608,23.8208)
    x_barwadda,y_barwadda = f_getPointCoordinates(root,86.4219,23.8554)
    x_kalyanpur,y_kalyanpur = f_getPointCoordinates(root,86.4083,23.8614)
    x_murradi,y_murradi = f_getPointCoordinates(root,86.4301,23.8663)
    x_kharni,y_kharni = f_getPointCoordinates(root,86.2844,23.8525)
    x_sonaria,y_sonaria = f_getPointCoordinates(root,86.3871,23.9069)
    x_tundipahar,y_tundipahar = f_getPointCoordinates(root,86.3934,23.9650)
    x_sonapani,y_sonapani = f_getPointCoordinates(root,86.3618,23.9420)
    x_jamkol,y_jamkol = f_getPointCoordinates(root,86.3947,23.9813)
    x_dhirajpur,y_dhirajpur = f_getPointCoordinates(root,86.3510,23.9796)
    x_bariarpur,y_bariarpur = f_getPointCoordinates(root,86.3413860056451,23.977514813080813)
    x_jitpur,y_jitpur = f_getPointCoordinates(root,86.3191442887595,23.975848628532525)
    x_wassepur,y_wassepur = f_getPointCoordinates(root,86.40298574280274,23.801774912233157)
    x_bankmore,y_bankmore = f_getPointCoordinates(root,86.416718556433,23.789367148379963)
    x_kusunda,y_kusunda = f_getPointCoordinates(root,86.37517659865782,23.776799121092047)
    x_katras,y_katras = f_getPointCoordinates(root,86.28995608194384,23.808413559855815)
    x_barakarriver,y_barakarriver = f_getPointCoordinates(root,86.45,23.99)
    x_gtroad,y_gtroad = f_getPointCoordinates(root,86.375,23.875)

    draw = ImageDraw.Draw(image)
    font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")
    draw.text((x_ism,y_ism), u"x IIT(ISM) DHANBAD", fill="#ffffff", font=font)
    draw.text((x_surya, y_surya), u"x SURYA HIGHLAND CITY", fill="#ffffff",font=font)
#    draw.text((x_stadium, y_stadium), u"x DHANBAD STADIUM", fill="#ffffff",font=font)
#    draw.text((x_airport,y_airport), u"x DHANBAD AIRPORT", fill="#ffffff",font=font)
    draw.text((x_birsa,y_birsa), u"x BIRSA MUNDA PARK", fill="#ffffff",font=font)
    draw.text((x_mohlidih,y_mohlidih), u"x MOHLIDIH", fill="#ffffff",font=font)
    draw.text((x_barwadda,y_barwadda), u"x BARWADDA GROUND", fill="#ffffff",font=font)
#    draw.text((x_kalyanpur,y_kalyanpur), u"x KALYANPUR GROUND", fill="#ffffff",font=font)
    draw.text((x_murradi,y_murradi), u"x MURRADI", fill="#ffffff",font=font)
    draw.text((x_kharni,y_kharni), u"x KHARNI", fill="#ffffff",font=font)
    draw.text((x_sonaria,y_sonaria), u"x SONARIA", fill="#ffffff",font=font)
    draw.text((x_tundipahar,y_tundipahar), u"x TUNDIPAHAR", fill="#ffffff",font=font)
    draw.text((x_sonapani,y_sonapani), u"x SONAPANI", fill="#ffffff",font=font)
    draw.text((x_jamkol,y_jamkol), u"x JAMKOL", fill="#ffffff",font=font)
#    draw.text((x_dhirajpur,y_dhirajpur), u"x DHIRAJPUR", fill="#ffffff",font=font)
    draw.text((x_bariarpur,y_bariarpur), u"x BARIARPUR", fill="#ffffff",font=font)
    draw.text((x_jitpur,y_jitpur), u"x JITPUR", fill="#ffffff",font=font)
    draw.text((x_barakarriver,y_barakarriver), u"BARAKAR RIVER", fill="#ffffff",font=font)
    draw.text((x_gtroad,y_gtroad), u"GT ROAD", fill="#ffffff",font=font)
    draw.text((x_wassepur,y_wassepur), u"x WASSEPUR", fill="#ffffff",font=font)
    draw.text((x_bankmore,y_bankmore), u"x BANK MORE", fill="#ffffff",font=font)
    draw.text((x_kusunda,y_kusunda), u"x KUSUNDA", fill="#ffffff",font=font)
    draw.text((x_katras,y_katras), u"x KATRAS", fill="#ffffff",font=font)
              
    return(image)

#Function to annotate plot
def f_annotatePlot(fig,ax,xs,ys):
    """
    Function to add data labels to plot.
    Inputs:
        fig: matplotlib figure
        ax: matplotlib axes
        xs: x-axis
        ys: y-axis
        
    Returns:
        fig: matplotlib figure
        ax: matplotlib axes
    """
    for x,y in zip(xs,ys):
        label = "{:.2f}".format(y)
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
    return(fig,ax)
    
#Function to make FCC image
def f_makeFCC(processed_image_list):
    """
    Function to make False Colour Composite Image
    
    Arguements:
        processed_image_list: list of processed image in each band
    Returns:
        None
    """
    green_band = Image.fromarray(np.uint8(processed_image_list[0]))
    red_band = Image.fromarray(np.uint8(processed_image_list[1]))
    nir_band = Image.fromarray(np.uint8(processed_image_list[2]))
    swir_band = Image.fromarray(np.uint8(processed_image_list[3]))
#    blue_band_arr = np.zeros([1153,1153,3],dtype=np.uint8)
#    blue_band_arr[:,:,2] = 255
#    blue_band_arr[:,:,1] = 255
#    blue_band = Image.fromarray(blue_band_arr).convert('L')
    
    #plt.imshow(blue_band)
#    empty_image_arr = np.zeros([1153,1153,3],dtype=np.uint8)
#    empty_image_arr[:] = 0
#    empty_image = Image.fromarray(empty_image_arr).convert('L')
    
    new_image_RGB = Image.merge('RGB',(nir_band,red_band,green_band))
    #Mark locations
    new_image_RGB = f_markLocations(new_image_RGB)
    new_image_RGB.save(f_getFilePath(f"reports/CompositeImage/Image{image_number}_FCC_RGB.tiff"))
    #Detect Edges
    new_image_RGB_edge = new_image_RGB.filter(ImageFilter.UnsharpMask(radius=2,percent=150, threshold=3))
    new_image_RGB_edge.save(f_getFilePath(f"reports/CompositeImage/EDGE_Image{image_number}_FCC_RGB.tiff"))
    
    new_image_CMYK = Image.merge('CMYK',(swir_band,nir_band,green_band,red_band))
    #Mark locations
    new_image_CMYK = f_markLocations(new_image_CMYK)
    new_image_CMYK.save(f_getFilePath(f"reports/CompositeImage/Image{image_number}_FCC_CMYK.tiff"))

#Function to calculate NDVI
def f_calculateNDVI(processed_image_list):
    """
    Function to calculate NDVI
    
    Arguements:
        processed_image_list: list of processed images in each band
    Returns:
        ndvi_green_arr: numpy array containing ndvi > 0.1
    """
    
    #NDVI: Normalized Difference Vegetation Index: (NIR-R)/(NIR+R)
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    # Calculate NDVI
    ndvi_arr = (processed_image_list[2].astype(float) - processed_image_list[1].astype(float)) / (processed_image_list[2] + processed_image_list[1])
    
    ndvi_blue_arr = ndvi_arr[:,:] < 0
    ndvi_blue = Image.fromarray(np.uint8(ndvi_blue_arr*255))
    #plt.imshow(ndvi_blue)
    #Mark locations
#    ndvi_blue = f_markLocations(ndvi_blue)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_blue_inanimate.tiff"),ndvi_blue,cmap="Blues")
    
    #Identify vegetation density
#    ndvi_green_arr = ndvi_arr[:,:] > 0.33
    ndvi_green_arr = ndvi_arr
    ndvi_green_arr[ndvi_green_arr < 0.1] = 0
    print(ndvi_green_arr.shape)
    ndvi_green = Image.fromarray(np.uint8(ndvi_green_arr*255))
    #Mark locations
#    ndvi_green = f_markLocations(ndvi_green)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_green_vegetation_density.tiff"),ndvi_green,cmap="RdYlGn")
    #plt.imshow(ndvi_green)
#    ndvi_green_dense_arr = ndvi_arr
#    ndvi_green_dense_arr[ndvi_green_dense_arr < 0.66] = 0
#    ndvi_green_sparse_arr = ndvi_arr
#    ndvi_green_sparse_arr[ndvi_green_sparse_arr > 0.66] = 0
#    ndvi_green_sparse_arr[ndvi_green_sparse_arr < 0.33] = 0
    
    ndvi_red_arr = ndvi_arr - (ndvi_blue_arr + ndvi_green_arr)
    ndvi_red = Image.fromarray(np.uint8(ndvi_red_arr*255))
    #Mark locations
#    ndvi_red = f_markLocations(ndvi_red)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_red_baresoil.tiff"),ndvi_red,cmap="Reds")
    #plt.imshow(ndvi_red)
    
    ndvi = Image.merge('RGB',(ndvi_red,ndvi_green,ndvi_blue))
#    ndvi = Image.fromarray(ndvi_arr).convert('L')
    #Mark locations
#    ndvi = f_markLocations(ndvi)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_RGB.tiff"),ndvi,cmap="hsv")
    #plt.imshow(ndvi)
    
    #NDWI: Normalized Difference Water Index: (Green-NIR)/(Green+NIR)
    ndwi_arr = (processed_image_list[0].astype(float) - processed_image_list[2].astype(float)) / (processed_image_list[0] + processed_image_list[2])
    ndwi = Image.fromarray(ndvi_arr).convert('L')
    #plt.imsave(f"reports/CompositeImage/Image_{image_number}/NDWI/ndwi.tiff",ndvi,cmap="inferno")
    #plt.imshow(ndwi)
    
    return(ndvi_green_arr,ndvi_arr)

#Function to calculate area under vegetation
def f_calculateArea(ndvi_green_arr, pixel_area,ndvi_arr,
                    total_vegetation_area_list,
                    dense_vegetation_area_list,
                    sparse_vegetation_area_list,
                    bare_soil_area_list,
                    very_healthy_vegetation_area_list,
                    mod_healthy_vegetation_area_list,
                    unhealthy_vegetation_area_list):
    """
    Function to calculate area under vegetation by category
    
    Arguements: 
        ndvi_green_arr: numpy array containing data of ndvi > 0.1
        pixel_area: area covered by a single pixel
        ndvi_arr
        total_vegetation_area_list,
        dense_vegetation_area_list,
        sparse_vegetation_area_list,
        bare_soil_area_list,
        very_healthy_vegetation_area_list,
        mod_healthy_vegetation_area_list,
        unhealthy_vegetation_area_list
    Returns:
        total_vegetation_area_list
        dense_vegetation_area_list
        sparse_vegetation_area_list
        bare_soil_area_list
        very_healthy_vegetation_area_list
        mod_healthy_vegetation_area_list
        unhealthy_vegetation_area_list
    """
    
    #Total vegetation area
    total_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.2) *pixel_area
    #Dense vegetation area
    dense_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.5) *pixel_area
    #Sparse vegetation area
    sparse_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.5) & (ndvi_green_arr > 0.2)) *pixel_area
    #Bare Soil area
    bare_soil_area = np.count_nonzero((ndvi_green_arr < 0.2) & (ndvi_green_arr > 0.1)) *pixel_area
    #Add to list of vegetation area by date
    dense_vegetation_area_list.append(dense_vegetation_area)
    sparse_vegetation_area_list.append(sparse_vegetation_area)
    total_vegetation_area_list.append(total_vegetation_area)
    bare_soil_area_list.append(bare_soil_area)
    #Percentage of vegetation
    percentage_total_vegetation = (total_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under vegetation : {round(total_vegetation_area,2)} sq.Km")
    print(f"Percentage area under vegetation : {round(percentage_total_vegetation,2)}%")
    percentage_dense_vegetation = (dense_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under dense vegetation : {round(dense_vegetation_area,2)} sq.Km")
    print(f"Percentage area under dense vegetation : {round(percentage_dense_vegetation,2)}%")
    percentage_sparse_vegetation = (sparse_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under sparse vegetation : {round(sparse_vegetation_area,2)} sq.Km")
    print(f"Percentage area under sparse vegetation : {round(percentage_sparse_vegetation,2)}%")
    percentage_bare_soil = (bare_soil_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under bare soil : {round(bare_soil_area,2)} sq.Km")
    print(f"Percentage area under bare soil : {round(percentage_bare_soil,2)}%")
    
    #Identify vegetation health
    ndvi_green_arr = ndvi_arr
    ndvi_green_arr[ndvi_green_arr < 0.2] = 0
    print(ndvi_green_arr.shape)
    ndvi_green = Image.fromarray(np.uint8(ndvi_green_arr*255))
    #Mark locations
#    ndvi_green = f_markLocations(ndvi_green)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_green_vegetation_health.tiff"),ndvi_green,cmap="RdYlGn")
    #Calculate area under vegetation
    #Very healthy vegetation area
    very_healthy_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.66) *pixel_area
    #Moderately healthy vegetation area
    mod_healthy_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.66) & (ndvi_green_arr > 0.33)) *pixel_area
    #Unhealthy vegetation area
#    unhealthy_vegetation_area_list = list(unhealthy_vegetation_area_list)
    unhealthy_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.33) & (ndvi_green_arr > 0.2)) *pixel_area
    print(unhealthy_vegetation_area)
    #Add to list of vegetation area by date
    print(type(unhealthy_vegetation_area_list))
    
    very_healthy_vegetation_area_list.append(very_healthy_vegetation_area)
    mod_healthy_vegetation_area_list.append(mod_healthy_vegetation_area)
    unhealthy_vegetation_area_list.append(unhealthy_vegetation_area)
    print(unhealthy_vegetation_area_list)
    #Percentage of vegetation
    percentage_vhealthy_vegetation = (very_healthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under very healthy vegetation : {round(very_healthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under very healthy vegetation : {round(percentage_vhealthy_vegetation,2)}%")
    percentage_mhealthy_vegetation = (mod_healthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under moderately healthy vegetation : {round(mod_healthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under moderately healthy vegetation : {round(percentage_mhealthy_vegetation,2)}%")
    percentage_unhealthy_vegetation = (unhealthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under unhealthy vegetation : {round(unhealthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under unhealthy vegetation : {round(percentage_unhealthy_vegetation,2)}%")
    
    return(total_vegetation_area_list,
           dense_vegetation_area_list,
           sparse_vegetation_area_list,
           bare_soil_area_list,
           very_healthy_vegetation_area_list,
           mod_healthy_vegetation_area_list,
           unhealthy_vegetation_area_list)
    
def f_populateDataframe(df_vegetation,
                        total_vegetation_area_list,
                        dense_vegetation_area_list,
                        sparse_vegetation_area_list,
                        bare_soil_area_list,
                        very_healthy_vegetation_area_list,
                        mod_healthy_vegetation_area_list,
                        unhealthy_vegetation_area_list):
    """
    Function to populate dataframe with vegetation data
    
    Arguements:
        df_vegetation: empty df
        total_vegetation_area_list,
       dense_vegetation_area_list,
       sparse_vegetation_area_list,
       bare_soil_area_list,
       very_healthy_vegetation_area_list,
       mod_healthy_vegetation_area_list,
       unhealthy_vegetation_area_list
    Returns:
        df_vegetation: populated df
    """
    df_vegetation['Date'] = image_date
    df_vegetation['Total_Vegetation_Area'] = total_vegetation_area_list
    df_vegetation['Dense_Vegetation_Area'] = dense_vegetation_area_list
    df_vegetation['Sparse_Vegetation_Area'] = sparse_vegetation_area_list
    df_vegetation['Bare_Soil_Area'] = bare_soil_area_list
    df_vegetation['VHealthy_Vegetation_Area'] = very_healthy_vegetation_area_list
    df_vegetation['MHealthy_Vegetation_Area'] = mod_healthy_vegetation_area_list
    df_vegetation['Unhealthy_Vegetation_Area'] = unhealthy_vegetation_area_list
    print("\n",df_vegetation.head(10))
    
    #Convert date values to datetime format
    df_vegetation["Date"] = pd.to_datetime(df_vegetation["Date"])
    df_vegetation = df_vegetation.sort_values(by=["Date"])
    #Save year and month as separate columns
    df_vegetation["Year"] = pd.DatetimeIndex(df_vegetation["Date"]).year
    df_vegetation["Month"] = pd.DatetimeIndex(df_vegetation["Date"]).month
    
    print(df_vegetation.info())
    
    return(df_vegetation)
        
#Function to plot vegetation data over time
def f_plotVeg(df_vegetation):
    """
    Function to plot vegetation time series data
    Arguements:
        df_vegetation: df containing vegetation data
    Returns: 
        None
    """

    #Plot vegetation density by date
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Date"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['Dense_Vegetation_Area'],label="Dense Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Dense_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['Sparse_Vegetation_Area'],label="Sparse Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Sparse_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['Bare_Soil_Area'],label="Bare Soil Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Bare_Soil_Area'])
    ax.set(xlabel="Date",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Density Over Four Years")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.autofmt_xdate()
    ax.grid()
    plt.savefig(f_getFilePath("reports/Plots/VegetationDensityByDatePlot"))
    plt.show()
    
    #Plot vegetation area by year
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Year"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['Dense_Vegetation_Area'],label="Dense Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Dense_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['Sparse_Vegetation_Area'],label="Sparse Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Sparse_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['Bare_Soil_Area'],label="Bare Soil Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Bare_Soil_Area'])
    ax.set(xlabel="Year",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Density Over Four Years")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.autofmt_xdate()
    ax.grid()
    plt.savefig(f_getFilePath("reports/Plots/VegetationDensityByYearPlot"))
    plt.show()
    
    #Plot vegetation area by month
    df_vegetation = df_vegetation.sort_values(by=["Month"])
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Month"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['Dense_Vegetation_Area'],label="Dense Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Dense_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['Sparse_Vegetation_Area'],label="Sparse Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Sparse_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['Bare_Soil_Area'],label="Bare Soil Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Bare_Soil_Area'])
    ax.set(xlabel="Month",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Density By Month")
    plt.setp(ax.get_xticklabels(), rotation=-30, horizontalalignment='right')
    fig.autofmt_xdate()
    ax.grid()
    ax.legend(loc="upper left")
    plt.savefig(f_getFilePath("reports/Plots/VegetationDensityByMonthPlot"))
    plt.show()
    
    #Plot vegetation health by date
    df_vegetation = df_vegetation.sort_values(by=["Date"])
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Date"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['VHealthy_Vegetation_Area'],label="Very Healthy Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['VHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['MHealthy_Vegetation_Area'],label="Moderately Healthy Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['MHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Date"],df_vegetation['Unhealthy_Vegetation_Area'],label="Unhealthy Vegetation Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Date"],df_vegetation['Unhealthy_Vegetation_Area'])
    ax.set(xlabel="Date",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Health Over Four Years")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.autofmt_xdate()
    ax.grid()
    plt.savefig(f_getFilePath("reports/Plots/VegetationHealthByDatePlot"))
    plt.show()
    
    #Plot vegetation health by year
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Year"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['VHealthy_Vegetation_Area'],label="Very Healthy Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['VHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['MHealthy_Vegetation_Area'],label="Moderately Healthy Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['MHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Year"],df_vegetation['Unhealthy_Vegetation_Area'],label="Unhealthy Vegetation Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Year"],df_vegetation['Unhealthy_Vegetation_Area'])
    ax.set(xlabel="Year",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Health Over Four Years")
    ax.legend(loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    fig.autofmt_xdate()
    
    ax.grid()
    plt.savefig(f_getFilePath("reports/Plots/VegetationHealthByYearPlot"))
    plt.show()
    
    #Plot vegetation health by month
    df_vegetation = df_vegetation.sort_values(by=["Month"])
    fig,ax = plt.subplots()
    ax.plot(df_vegetation["Month"],df_vegetation['Total_Vegetation_Area'],label="Total Vegetation Area",color='blue',marker='*',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Total_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['VHealthy_Vegetation_Area'],label="Very Healthy Vegetation Area",color='green',marker='+',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['VHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['MHealthy_Vegetation_Area'],label="Moderately Healthy Vegetation Area",color='orange',marker='o',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['MHealthy_Vegetation_Area'])
    ax.plot(df_vegetation["Month"],df_vegetation['Unhealthy_Vegetation_Area'],label="Unhealthy Vegetation Area",color='brown',marker='^',linewidth=2)
    fig,ax = f_annotatePlot(fig,ax,df_vegetation["Month"],df_vegetation['Unhealthy_Vegetation_Area'])
    ax.set(xlabel="Month",ylabel="Area Under Vegetation (sq.km)",title="Vegetation Health By Month")
    plt.setp(ax.get_xticklabels(), rotation=-30, horizontalalignment='right')
    fig.autofmt_xdate()
    ax.grid()
    ax.legend(loc="upper left")
    plt.savefig(f_getFilePath("reports/Plots/VegetationHealthByMonthPlot"))
    plt.show()
    
#Function to run correlation
def f_correlation(df_vegetation):
    """
    Function to run correlation on vegetation data
    Arguements:
        df_vegetation: df containing vegetation data
    Returns:
        None
    """    
    print("\nCorrelation between Total Vegetation Area and Month")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["Month"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Dense Vegetation Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["Dense_Vegetation_Area"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Sparse Vegetation Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["Sparse_Vegetation_Area"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Bare Soil Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["Bare_Soil_Area"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Very Healthy Vegetation Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["VHealthy_Vegetation_Area"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Mod Healthy Vegetation Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["MHealthy_Vegetation_Area"])[0,1])
    print("\nCorrelation between Total Vegetation Area and Unhealthy Vegetation Area")
    print(np.corrcoef(df_vegetation["Total_Vegetation_Area"],df_vegetation["Unhealthy_Vegetation_Area"])[0,1])
    
    corr_matrix = df_vegetation.corr()
    print("\n",corr_matrix)
    ax = sn.heatmap(corr_matrix, annot=True)
    ax.figure.tight_layout()
    plt.savefig(f_getFilePath("reports/plots/CorrelationMatrix"))
    plt.show()
    
#Function to run regression models
def f_regModels(df_vegetation):
    """Function to run regression models on vegetation data
    
    Arguements:
        df_vegetation: df containing vegetation
    Return:
        None
    """
    #Regress total vegetation area over month
    print("\nRegression 1")
    Y = df_vegetation["Total_Vegetation_Area"]
    X = df_vegetation["Month"]
    #X= sm.add_constant(X)
    mod = sm.OLS(Y,X)
    res = mod.fit()
    print(res.summary())
    Yhat = res.predict(X)
    print("\nPredicted Values\n",Yhat)
    #Plot observed vs estimated values
    fig,ax = plt.subplots()
    ax.scatter(X,Y,label="Observed",color="blue",marker="+")
    ax.plot(X,Yhat,label="Estimated",color="red",marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Vegetation Area")
    ax.set_title("Observed vs Estimated Values")
    ax.legend(loc="lower right")
    #slope = (Yhat[1]-Yhat[0])/(X[1]-X[0])
    #angle = math.atan(slope)
    #ax.annotate(f"Y={res.params[0]}*X",
    #                 (((df_vegetation["Month"].max()-df_vegetation["Month"].min())/2),
    #                  ((df_vegetation["Total_Vegetation_Area"].max()-df_vegetation["Total_Vegetation_Area"].min())/2)),
    #                  ha="left",
    #                  rotation=angle)
    plt.savefig(f_getFilePath("reports/plots/MonthVsTotalRegression1Plot"))
    plt.show()
    #Plot observed, estimated and predicted values
    Xnew = pd.Series([4,5,6,7,8])
    Yphat = res.predict(Xnew)
    fig,ax = plt.subplots()
    ax.scatter(X,Y,label="Observed",color="green",marker="+")
    ax.scatter(X,Yhat,label="Estimated",color="red",marker="o")
    ax.scatter(Xnew,Yphat,label="Predicted",color="blue",marker="^")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Vegetation Area")
    ax.set_title("Observed vs Predicted Values")
    ax.legend(loc="lower right")
    plt.savefig(f_getFilePath("reports/plots/MonthVsTotalPredictionPlot"))
    plt.show()    
    
    #Regress total vegetation over month, dense vegetation
    print("\nRegression 2")
    Y = df_vegetation["Total_Vegetation_Area"]
    X = df_vegetation[["Dense_Vegetation_Area"]]
    #X= sm.add_constant(X)
    mod = sm.OLS(Y,X)
    res = mod.fit()
    print(res.summary())
    #params = res.params
    Yhat = res.predict(X)
    print("\nPredicted Values",Yhat)
    df_vegetation = df_vegetation.sort_values(by=["Dense_Vegetation_Area"])
    #Plot observed vs estimated values
    fig,ax = plt.subplots()
    ax.scatter(X,Y,label="Observed",color="blue",marker="+")
    ax.plot(X,Yhat,label="Estimated",color="red",marker="o")
    ax.set_xlabel("Dense Vegetation Area")
    ax.set_ylabel("Total Vegetation Area")
    ax.set_title("Observed vs Estimated Values")
    ax.legend(loc="lower right")
    #print("Yhat[1]: ",type(Yhat))
    #slope = (Yhat[1]-Yhat[0])/(X[1]-X[0])
    #angle = math.atan(slope)
    #ax.annotate(f"Y={res.params[0]}*X",
    #                 (((df_vegetation["Dense_Vegetation_Area"].max()-df_vegetation["Dense_Vegetation_Area"].min())/2),
    #                  ((df_vegetation["Total_Vegetation_Area"].max()-df_vegetation["Total_Vegetation_Area"].min())/2)),
    #                  ha="left",
    #                  rotation=angle)
    plt.savefig(f_getFilePath("reports/plots/DenseVsTotalRegression2Plot"))
    
    plt.show()
    
    #Regress total vegetation over month, very healthy vegetation, unhealthy vegetation
    print("\nRegression 3")
    Y = df_vegetation["Total_Vegetation_Area"]
    X = df_vegetation[["VHealthy_Vegetation_Area","Unhealthy_Vegetation_Area"]]
    #X= sm.add_constant(X)
    mod = sm.OLS(Y,X)
    res = mod.fit()
    print(res.summary())
    print("\nPredicted Values\n",res.predict(X))

#Main function
              
#Initialize variables
processed_image_list = []
total_images = 4
total_vegetation_area_list = []
dense_vegetation_area_list = []
sparse_vegetation_area_list = []
bare_soil_area_list = []
very_healthy_vegetation_area_list = []
mod_healthy_vegetation_area_list = []
unhealthy_vegetation_area_list = []
spatial_resolution_deg = []
image_date = []
df_vegetation = pd.DataFrame()

print(type(unhealthy_vegetation_area_list))
    

#Loop through each image
for image_number in range(1,total_images+1):
    #Get metadata
    # create element tree object 
    for xmlfile in glob.iglob(f_getFilePath(f'data/*_Image{image_number}/*.xml')):
        tree = ET.parse(xmlfile) 
    # get root element 
    root = tree.getroot() 
    #iterate through metadata
    for image_data in root.findall('For_Image_Data'):
        print(image_data)
        for desc_item in image_data:
            print(desc_item)
            if desc_item.tag == "Date_of_Pass":
                image_date.append(desc_item.text)
            elif desc_item.tag == 'Spatial_Resolution':
                spatial_resolution_deg.append(float(desc_item.text.split()[0]))
            else:
                continue
    
    #Inspect original image and make histogram
    band_number = 2
#    hfig, hax = plt.subplots(2,2,sharex=True,sharey=True)
#    hax = hax.ravel()
    for filepath in glob.iglob(f_getFilePath(f'data/*_Image{image_number}/*.tif*')):
    #    print(filepath)
        image_iBand = imageio.imread(filepath)
        cmap = f_chooseColourMap(band_number)
        plt.imsave(f_getFilePath(f"reports/OriginalInColour/Image{image_number}_Band{band_number}.jpg"), image_iBand, cmap=cmap)
        print(f"\nInspect Original Image {image_number} in Band {band_number}...")
#        hfig, hax = f_plotHistogram(hfig,hax,image_iBand, band_number, image_number)
        band_number = band_number + 1
#    hfig.suptitle(f"Histogram of DN values of Original image {image_number}")
#    hfig.subplots_adjust(top=0.88)
#    plt.savefig(f_getFilePath(f"reports/histograms/OriginalImage_{image_number}"))
##    plt.show()
    
    #Process image and make histogram
    band_number = 2
    processed_image_list.clear()
    hfig, hax = plt.subplots(2,2,sharex=True,sharey=True)
    hax = hax.ravel()
    for filepath in glob.iglob(f_getFilePath(f'data/*_Image{image_number}/*.tif*')):
    #    print(filepath)
        image_iBand = imageio.imread(filepath)
        print(f"\nProcessing Image {image_number} in Band {band_number}...")
        cmap = f_chooseColourMap(band_number)
        processed_image, hfig, hax = f_processImage(image_iBand, band_number, image_number, hfig, hax, cmap)
        processed_image_list.append(processed_image)
        band_number = band_number + 1
#    hfig.suptitle(f"Histogram of DN values of Rescaled and Equalized image {image_number}")
#    hfig.subplots_adjust(top=0.88)
#    plt.savefig(f_getFilePath(f"reports/histograms/RescaledEqualizedImage_{image_number}"))
#    plt.show()
#    
    #Create False Colour Composite Image
    f_makeFCC(processed_image_list)

    #Calculate NDVI
    ndvi_green_arr,ndvi_arr = f_calculateNDVI(processed_image_list)
    
    #Calculate area under vegetation
    spatial_resolution = spatial_resolution_deg[image_number - 1]*111*1000
    pixel_area = (spatial_resolution*spatial_resolution)/1000000
    print(type(unhealthy_vegetation_area_list))
#    total_vegetation_area_list, dense_vegetation_area_list,
#    sparse_vegetation_area_list, bare_soil_area_list,
#    very_healthy_vegetation_area_list, mod_healthy_vegetation_area_list,
#    unhealthy_vegetation_area_list = f_calculateArea(ndvi_green_arr, pixel_area,ndvi_arr,
#                                                     total_vegetation_area_list,
#                                                     dense_vegetation_area_list,
#                                                     sparse_vegetation_area_list,
#                                                     bare_soil_area_list,
#                                                     very_healthy_vegetation_area_list,
#                                                     mod_healthy_vegetation_area_list,
#                                                     unhealthy_vegetation_area_list)
    #Total vegetation area
    total_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.2) *pixel_area
    #Dense vegetation area
    dense_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.5) *pixel_area
    #Sparse vegetation area
    sparse_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.5) & (ndvi_green_arr > 0.2)) *pixel_area
    #Bare Soil area
    bare_soil_area = np.count_nonzero((ndvi_green_arr < 0.2) & (ndvi_green_arr > 0.1)) *pixel_area
    #Add to list of vegetation area by date
    dense_vegetation_area_list.append(dense_vegetation_area)
    sparse_vegetation_area_list.append(sparse_vegetation_area)
    total_vegetation_area_list.append(total_vegetation_area)
    bare_soil_area_list.append(bare_soil_area)
    #Percentage of vegetation
    percentage_total_vegetation = (total_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under vegetation : {round(total_vegetation_area,2)} sq.Km")
    print(f"Percentage area under vegetation : {round(percentage_total_vegetation,2)}%")
    percentage_dense_vegetation = (dense_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under dense vegetation : {round(dense_vegetation_area,2)} sq.Km")
    print(f"Percentage area under dense vegetation : {round(percentage_dense_vegetation,2)}%")
    percentage_sparse_vegetation = (sparse_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under sparse vegetation : {round(sparse_vegetation_area,2)} sq.Km")
    print(f"Percentage area under sparse vegetation : {round(percentage_sparse_vegetation,2)}%")
    percentage_bare_soil = (bare_soil_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under bare soil : {round(bare_soil_area,2)} sq.Km")
    print(f"Percentage area under bare soil : {round(percentage_bare_soil,2)}%")
    
    #Identify vegetation health
    ndvi_green_arr = ndvi_arr
    ndvi_green_arr[ndvi_green_arr < 0.2] = 0
    print(ndvi_green_arr.shape)
    ndvi_green = Image.fromarray(np.uint8(ndvi_green_arr*255))
    #Mark locations
#    ndvi_green = f_markLocations(ndvi_green)
    plt.imsave(f_getFilePath(f"reports/CompositeImage/Image{image_number}_ndvi_green_vegetation_health.tiff"),ndvi_green,cmap="RdYlGn")
    #Calculate area under vegetation
    #Very healthy vegetation area
    very_healthy_vegetation_area = np.count_nonzero(ndvi_green_arr > 0.66) *pixel_area
    #Moderately healthy vegetation area
    mod_healthy_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.66) & (ndvi_green_arr > 0.33)) *pixel_area
    #Unhealthy vegetation area
#    unhealthy_vegetation_area_list = list(unhealthy_vegetation_area_list)
    unhealthy_vegetation_area = np.count_nonzero((ndvi_green_arr < 0.33) & (ndvi_green_arr > 0.2)) *pixel_area
    print(unhealthy_vegetation_area)
    #Add to list of vegetation area by date
    print(type(unhealthy_vegetation_area_list))
    
    very_healthy_vegetation_area_list.append(very_healthy_vegetation_area)
    mod_healthy_vegetation_area_list.append(mod_healthy_vegetation_area)
    unhealthy_vegetation_area_list.append(unhealthy_vegetation_area)
    print(unhealthy_vegetation_area_list)
    #Percentage of vegetation
    percentage_vhealthy_vegetation = (very_healthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under very healthy vegetation : {round(very_healthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under very healthy vegetation : {round(percentage_vhealthy_vegetation,2)}%")
    percentage_mhealthy_vegetation = (mod_healthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under moderately healthy vegetation : {round(mod_healthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under moderately healthy vegetation : {round(percentage_mhealthy_vegetation,2)}%")
    percentage_unhealthy_vegetation = (unhealthy_vegetation_area/(ndvi_green_arr.size*pixel_area))*100
    print(f"\nArea under unhealthy vegetation : {round(unhealthy_vegetation_area,2)} sq.Km")
    print(f"Percentage area under unhealthy vegetation : {round(percentage_unhealthy_vegetation,2)}%")
    
    print(type(unhealthy_vegetation_area_list))
    
    
#Populate dataframe
print(mod_healthy_vegetation_area_list)
df_vegetation = f_populateDataframe(df_vegetation,
                                    total_vegetation_area_list,
                                    dense_vegetation_area_list,
                                    sparse_vegetation_area_list,
                                    bare_soil_area_list,
                                    very_healthy_vegetation_area_list,
                                    mod_healthy_vegetation_area_list,
                                    unhealthy_vegetation_area_list)

#Plot time series of vegetation 
f_plotVeg(df_vegetation)

#Correlation
f_correlation(df_vegetation)

#Regression Models
f_regModels(df_vegetation)