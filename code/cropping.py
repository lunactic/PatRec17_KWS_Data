from xml.dom import minidom
import numpy as np
import re
from scipy import ndimage
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import image

#Based on https://stackoverflow.com/questions/21566610/crop-out-partial-image-using-numpy-or-scipy
def masking(number):
    img = image.imread('../images/%d.jpg'% number)

    doc = minidom.parse('../ground-truth/locations/%d.svg'% number) 

    masks={}
    result={}
    for path in doc.getElementsByTagName('path'):
        _d = path.getAttribute('d')
        _d = re.sub('[MLSZ]','',_d)
        d = np.fromstring(_d, dtype=float, sep=' ')
        d = np.reshape(d,(int(len(d)/2),2))
        masks[path.getAttribute('id')] = d
    doc.unlink()
    
    for i in masks:
        cell = masks[i]
        pth = Path(cell, closed=False)

        xc = cell[:,0]
        yc = cell[:,1]
                    
        nr, nc = img.shape
        ygrid, xgrid = np.mgrid[:nr, :nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

        mask = pth.contains_points(xypix)

        mask = mask.reshape(img.shape)

        masked = np.ma.masked_array(img, ~mask)


        xmin, xmax = int(xc.min()), int(np.ceil(xc.max()))
        ymin, ymax = int(yc.min()), int(np.ceil(yc.max()))
        trimmed = masked[ymin:ymax, xmin:xmax]
        result[i]=trimmed
        
        #testing print the different words
        # imgplot = plt.imshow(trimmed, cmap=plt.cm.gray)
        # plt.show()
    return result

#TODO: return the id for comparison