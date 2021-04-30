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
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
from pydic import Display
from scipy import stats
# locate the pydic module and import it
import imp
import pydic

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:  # save image
        cv2.imwrite(filepath + "image" + str(count) + ".jpg", image)
    return hasFrames


# pydic = imp.load_source('pydic', 'C:/Users/Matthew/Documents/GitHub/DIC/pydic.py')
disp = Display()
'''
com = disp.getText('Please enter the correct COM port', 'COM4')
testChoice = disp.chooseTest(com)
tp = 'none'

if testChoice == 'three':
    tp = disp.tpType()
if testChoice == 'three' and tp == 'rect' or testChoice == 'tensile':
    width = float(disp.getText('Please enter the specimen width in millimeters', '10.0'))
    thic = float(disp.getText('Please enter the specimen thickness in millimeters', '3.1'))
if testChoice == 'three' and tp == 'circ':
    diam = float(disp.getText('Please enter the specimen diameter in millimeters', '3.0'))
'''
disp.recording()
seconds, psi, vidName, diam, height, width, thic, ind, testChoice = disp.dataOut()
# filepath = disp.getFile()
filesplit = vidName.split('/')
filepath = ''
del filesplit[-1]
for i in range(len(filesplit)):
    filepath = filepath + filesplit[i] + '/'
psi = np.array(psi)
A = 2*math.pi*(25/25.4)**2  # piston bore area in inches
a = 2*math.pi*(10/25.4)**2  # piston neck area in inches

if testChoice == 'tensile':
    force = (psi*A)*4.44822  # convert psi to Newtons
elif testChoice == 'hard' or 'tprect' or 'tpcirc':
    force = (psi*(A-a))*4.44822  # convert psi to Newtons
if testChoice == 'hard':
    diam = float(disp.getText('Please enter the indent diameter in millimeters', '0.1'))
elif testChoice == 'tprect' or 'tpcirc' or 'tensile':
 #   vidName = disp.getText("Video File Name:", "PS1.mp4")
 #   disp.closeData()

    # turn imported video into pictures

    vidcap = cv2.VideoCapture(vidName)

    sec = 0
    for i in range(len(seconds)):
        count = 1001 + i
        sec = seconds[i]
        sec = round(sec, 2)
        getFrame(sec)

    #  ====== RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS (STRUCTURED GRID)
    correl_wind_size = (80, 80)  # the size in pixel of the correlation windows
    correl_grid_size = (20, 20)  # the size in pixel of the interval (dx,dy)
    # of the correlation grid
    pydic.init(filepath + '*.jpg', correl_wind_size,
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
elif testChoice == 'tprect' or 'tpcirc':
    L = 4.5*25.4  # L is the distance between the supports
    if testChoice == 'tprect':  # rectangluar cross section
        # compute the maximal normal stress with this force
        # the maximal normal stress is located in the lower plane
        max_stress = (1.5) * force * L / (width * thic ** 2)  # stress in MPa
    elif testChoice == 'tpcirc':  # circular cross section
        max_stress = force * L / (math.pi*((diam/2) ** 3))  # stress in MPa
    # now extract the maximal average strains on the lower plane of the sample
    x_range = range(1, pydic.grid_list[0].size_x - 1)
    y_range = range(pydic.grid_list[0].size_y - 1, pydic.grid_list[0].size_y)

    # - use grid.average method to compute the average values of the xx and yy strains
    max_strain_xx = np.array([grid.average(grid.strain_xx, x_range, y_range) for grid in pydic.grid_list])
    strain = max_strain_xx.tolist()
if testChoice == 'hard':
    load = (max(force))/9.80665  # get load in kg
    D = (7/32)*25.4  # ball bearing diam in mm
    # calculate Brinnell Hardness
    hardness = load/(math.pi*D/2*(D-math.sqrt(D**2-diam**2)))
    disp.saveData(filepath, force=max(force), diam=diam, hardness=hardness)
elif testChoice == 'tprect' or 'tpcirc' or 'tensile':

    force = force.tolist()
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
    stress = force/(width * thic)  # stress in MPa
    disp = Display()
    disp.dataPlot(stress, ave_strain_yy, seconds, force)
    disp.saveData(filepath, seconds=seconds, force=force, stress=stress, strain=strain)
if testChoice == 'tensile':
    # compute Young's modulus with scipy linear regression
    E, intercept, r_value, p_value, std_err = stats.linregress(ave_strain_yy, stress)
    # compute Poisson's ratio with scipy linear regression
    Nu, intercept, r_value, p_value, std_err = stats.linregress(ave_strain_yy, -ave_strain_xx)

    print("\nThe computed elastic constants are :")
    print("  => Young's modulus E={:.2f} GPa".format(E*1e-9))
    print("  => Poisson's ratio Nu={:.2f}".format(Nu))
