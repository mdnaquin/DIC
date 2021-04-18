#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of pydic, a free digital correlation suite for computing
# strain fields
#
# Author :  - Damien ANDRE, SPCTS/ENSIL-ENSCI, Limoges France
#             <damien.andre@unilim.fr>
#
# Copyright (C) 2017 Damien ANDRE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ====== INTRODUCTION
# The tensile example shows how to use the pydic module to compute
# the young's modulus and the poisson's ratio from picture captured
# during a tensile test. The loading were recorded during the test
# and the values are stored in the meta-data file (see 'img/meta-data.txt').
#
# Note that :
#  - pictures of the tensile test are located in the 'img' directory
#  - for a better undestanding, please refer to the 'description.png' file
#    that describes the tensile test


# ====== IMPORTING MODULES
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import cv2
from pydic import Display
import math
# locate the pydic module and import it
import imp


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(filepath+"/image"+str(count)+".jpg", image)  # save image
    return hasFrames


pydic = imp.load_source('pydic', 'C:/Users/Matthew/Documents/GitHub/DIC/pydic.py')
disp = Display()
com = disp.getText('Please enter the correct COM port', 'COM5')
testChoice = disp.chooseTest(com)
tp = 'none'

if testChoice == 'three':
    tp = disp.tpType()
if testChoice == 'three' and tp == 'rect' or testChoice == 'tensile':
    width = float(disp.getText('Please enter the specimen width in inches', '0.394'))
    thic = float(disp.getText('Please enter the specimen thickness in inches', '0.125'))
if testChoice == 'three' and tp == 'circ':
    diam = float(disp.getText('Please enter the specimen diameter in inches', '0.125'))

disp.recording()
seconds, psi = disp.dataOut()
filepath = disp.getText("Filepath:", "C:/Users/mdnaq_000/Downloads/DIC/PS1")
psi = np.array(psi)

if testChoice == 'tensile':
    force = psi/((50/25.4)**2)
elif testChoice == 'hard' or 'three':
    force = psi/(((50-20)/25.4)**2)
if testChoice == 'hard':
    diam = float(disp.getText('Please enter the indent diameter in inches', '0.01'))
elif testChoice == 'three' or 'tensile':
    vidName = disp.getText("Video File Name:", "PS1.mp4")
    disp.closeData()

    # turn imported video into pictures

    vidcap = cv2.VideoCapture(filepath+"/"+vidName)

    sec = 0
    frameRate = 1  # //it will capture image in each X seconds
    count = 1001
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

    '''
    for i in range(len(seconds)):
        count = 1001 + i
        sec = seconds[i]
        sec = round(sec, 2)
        getFrame(sec)
    '''

    #  ====== RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS (STRUCTURED GRID)
    correl_wind_size = (80, 80)  # the size in pixel of the correlation windows
    correl_grid_size = (20, 20)  # the size in pixel of the interval (dx,dy)
    # of the correlation grid
    '''
    # read image series and write a separated result file
    pydic.init(filepath+'/*.bmp', correl_wind_size,
               correl_grid_size, "result.dic")


    # and read the result file for computing strain and displacement field
    # from the result file

    pydic.read_dic_file('result.dic', interpolation='spline', strain_type='cauchy',
                        save_image=True, scale_disp=10, scale_grid=25,
                        meta_info_file='D:/College/MEEN 401-402/DIC/meta-data.txt')


    #  ====== OR RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS
    # (WITH UNSTRUCTURED GRID OPTION)
    # note that you can't use the 'spline' or the 'raw' interpolation with
    # unstructured grids please uncomment the next lines if you want to use the
    # unstructured grid options instead of the aligned grid
    '''

    pydic.init(filepath+'/*.jpg', correl_wind_size,
               correl_grid_size, "result.dic", unstructured_grid=(20, 5))

    pydic.read_dic_file('result.dic', interpolation='cubic', save_image=True,
                        scale_disp=10, scale_grid=25)

    #  ====== RESULTS
    # Now you can go in the 'img/pydic' directory to see the results :
    # - the 'disp', 'grid' and 'marker' directories contain image files
    # - the 'result' directory contain raw text csv file where displacement
    # and strain fields are written

    # ======= STANDARD POST-TREATMENT : STRAIN FIELD MAP PLOTTING
    # the pydic.grid_list is a list that contains all the correlated grids
    # (one per image)
    # the grid objects are the main objects of pydic
    last_grid = pydic.grid_list[-1]
    last_grid.plot_field(last_grid.strain_xx, 'xx strain')
    last_grid.plot_field(last_grid.strain_yy, 'yy strain')
    plt.show()


    # now extract the main average strains on xx and yy
    # - first, we need to reduce the interest zone where the
    # average values are computed
if testChoice == 'tensile':
    test = pydic.grid_list[0].size_x/4

    x_range = range(int(pydic.grid_list[0].size_x/4),
                    int(3*pydic.grid_list[0].size_x/4))
    y_range = range(int(pydic.grid_list[0].size_y/4),
                    int(3*pydic.grid_list[0].size_y/4))
    # - use grid.average method to compute the average values
    # of the xx and yy strains
    ave_strain_xx = np.array([grid.average(grid.strain_xx, x_range, y_range)
                              for grid in pydic.grid_list])
    ave_strain_yy = np.array([grid.average(grid.strain_yy, x_range, y_range)
                              for grid in pydic.grid_list])
    strain = ave_strain_yy.tolist()
    strainlen = len(ave_strain_yy)
elif testChoice == 'threept':
    # compute the maximal normal stress with this force
    # the maximal normal stress is located in the lower plane
    L = 4.5  # L is the higher distance between the supports
    max_stress = (3. / 2.) * force * L / (width * thic ** 2)

    # now extract the maximal average strains on the lower plane of the sample along
    x_range = range(1, pydic.grid_list[0].size_x - 1)
    y_range = range(pydic.grid_list[0].size_y - 1, pydic.grid_list[0].size_y)

    # - use grid.average method to compute the average values of the xx and yy strains
    max_strain_xx = np.array([grid.average(grid.strain_xx, x_range, y_range) for grid in pydic.grid_list])
    strain = max_strain_xx.tolist()
if testChoice == 'hard':
    force = max(force)
    D = 7/32
    hardness = force/(math.pi*D/2*(D-math.sqrt(D**2-diam**2)))
    print(hardness)
    disp.saveData(filepath, lb=force, d=diam, hardness=hardness)
elif testChoice == 'three' or 'tensile':
    '''
    forcelen = len(force)
    if forcelen > strainlen:
        diff = forcelen - strainlen
        for i in range(diff):
            del force[-1]
            del seconds[-1]
    elif strainlen > forcelen:
        diff = strainlen - forcelen
        for i in range(diff):
            del ave_strain_yy[-1]
    
    force = np.array(force)
    stress = force/(width * thic)
    disp = Display(com)
    disp.dataPlot(stress, ave_strain_yy, seconds, force)
    '''
    force = np.zeros(strainlen)
    stress = np.zeros(strainlen)
    seconds = np.linspace(0, sec, (strainlen+1))

    disp.saveData(filepath, seconds=seconds, force=force, stress=stress, strain=strain)


'''
# compute the maximal normal stress with this force
# the maximal normal stress is located in the lower plane
L = (41. - 5.)*1e-3 # L is the higher distance between the supports
l = (19. - 5.)*1e-3 # l is the higher distance between the supports
b = 7.66e-3    # b is the sample width
h = 4.06e-3    # h is the sample thickness
max_stress = (3./2.)* force *(L-l)/(b*h**2)


# now extract the maximal average strains on the lower plane of the sample along
x_range = range(1                          , pydic.grid_list[0].size_x-1)
y_range = range(pydic.grid_list[0].size_y-1, pydic.grid_list[0].size_y  )

# - use grid.average method to compute the average values of the xx and yy strains
max_strain_xx = np.array([grid.average(grid.strain_xx, x_range, y_range) for grid in pydic.grid_list])
strain = max_strain_xx.tolist()
# now compute Young's modulus thanks to scipy linear regression
E, intercept, r_value, p_value, std_err = stats.linregress(max_strain_xx, max_stress)

# and print results !
print ("\nThe computed elastic constant is :")
print ("  => Young's modulus E={:.2f} GPa".format(E*1e-9))
'''