import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def extractBoundary(img_name):
    #current extraction from images unable to find holes within image
    #could potentially use contour hierarchies (still need to distinguish between buildings and farm)
    #define within farm object areas of non sampling
    img_path = "./"+img_name
    split = img_name.split(".")
    img_name = split[0]
    file_type = split[1]

    og = cv2.imread(img_path)
    img = cv2.imread(img_path, 0)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(img)

    denoised = denoiseContours(contours)

    cv2.drawContours(out, denoised, -1, 255, 1)
    """
    cv2.imshow('Original', og)
    cv2.imshow('Boundary', out)
    """
    result_path = "./" + img_name + "_results." + file_type
    cv2.imwrite(result_path, out)
    cv2.waitKey(0)

    finalBoundary = denoised

    return finalBoundary

def denoiseContours(contours):
    #only want contours that have area above threshold
    #just taking max area for now
    max_area_idx = np.argmax([cv2.contourArea(contour) for contour in contours])
    return contours[max_area_idx]

def constructFarm(img_path, crop):
    boundary = extractBoundary(img_path)
    crop_boundary = extractBoundary(img_path)
    crop_areas = {crop: crop_boundary}
    return Farm(boundary, img_path, crop_areas)

class Farm:
    """
    Represents a Farm
    """
    #dictionary of coords to plots
    plots = {}
    #array of 2D array of points ie: [[[0, 0]], [[1, 1]]]
    boundary = None
    bounding_box = None
    img_path = None
    #dictionary of boundaries ie: {corn: [[[0, 0]], [[1, 1]]]}
    crop_areas = None
    #create some overlying mesh for the elevation of the farm
    #ie: [[12, 12, 12], [13, 13, 13]] for 2x3 farm
    #each "coord" (index) corresponds to an elevation here
    elevation_mesh = []
    coord_mask = []
    X_mesh, Y_mesh = None, None

    #crop areas should be within boundary for when instantiating plots
    #only add for crop areas within farm boundary
    def __init__(self, boundary, img_path = None, crop_areas = None):
        #right now boundary not translated to real world just random coordinates
        #TODO: translate to some long/lat scheme - add some sort of mapping
        self.boundary = boundary
        x, y, w, h = cv2.boundingRect(boundary)
        self.bounding_box = [[x, y], [x, y + h], [x + w, y], [x + w, y + h]]
        X = np.arange(x, x + w + 1, 1)
        Y = np.arange(y, y + h + 1, 1)
        self.X_mesh, self.Y_mesh = np.meshgrid(X, Y)
        self.img_path = img_path
        self.crop_areas = crop_areas
        self.elevation_mesh = np.empty((w + 1, h + 1,))
        self.elevation_mesh[:] = 0
        self.coord_mask = np.empty((w + 1, h + 1,))
        #used for visualization masking purposes only
        #if point in farm then coord_mask set to 0
        self.coord_mask[:] = np.nan
        self.plots = self.define_points_in_farm()

    def __str__(self):
        return f"{self.boundary})"


    def point_in_farm(self, point):
        #either ray implementation or cv2
        in_farm = cv2.pointPolygonTest(self.boundary, point, measureDist = False)
        if in_farm < 0:
            return False
        return True

    def point_in_crop_area(self, point):
        for area, boundary in self.crop_areas.items():
            in_crop_area = cv2.pointPolygonTest(boundary, point, measureDist = False)
            if in_crop_area == 1:
                return area
        return None

    def define_points_in_farm(self):
        plots = {}
        x, y, width, height = cv2.boundingRect(self.boundary)
        for ii in range(x, x + width + 1):
            for jj in range(y, y + height + 1):
                if self.point_in_farm((ii, jj)):
                    crop = self.point_in_crop_area((ii, jj))
                    newPlot = Plot([[ii, jj]], crop)
                    #first set everything within farm to 0
                    #ii jj not 0 indexed need to "0 index" to translate into
                    #bounding box for np array
                    start_x = self.bounding_box[0][0]
                    start_y = self.bounding_box[0][1]
                    self.coord_mask[ii - start_x][jj - start_y] = 0
                    plots[(ii, jj)] = newPlot
        return plots

    def visualize(self):
        #plot boundary
        #for each plot in farm
        #3d gradient of carbon level + also elevation
        #TODO: FIX LATER plot only where the farm mask is not nan
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        x = []
        y = []
        z = []
        for plot in self.plots.values():
            x.append(plot.coord[0][0])
            y.append(plot.coord[0][1])
            z.append(plot.carbon)
            #ax.scatter(plot.coord[0][0], plot.coord[0][1], plot.carbon)

        surf = ax.plot_trisurf(x, y, z, linewidth=0.1)

        ax.set_title("SOC Levels in Farm")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("SOC Level")

        plt.show()

    def visualize_elev(self):

        #apply mask np array only plot parts of surface where z not nan (ie: points in farm)
        masked_elev = np.ma.masked_where(np.isnan(self.coord_mask), self.elevation_mesh)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.X_mesh, self.Y_mesh, masked_elev.T, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #TODO: make the z axis less crazy please
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Peat Height')
        #ax.set_zlim3d(0, 200)
        plt.show()

class Plot:
    """
    Sampleable plot of land within the farm
    """
    latitude = None
    longitude = None
    slope_y = 0
    slope_x = 1
    elevation = 10
    crop = None
    coord = [[0, 0]]
    carbon = 0

    def __init__(self, coord, crop):
        self.coord = coord
        self.crop = crop

    def __str__(self):
        return str(self.coord) + str(self.crop)
