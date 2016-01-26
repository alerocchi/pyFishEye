import numpy as np
from cv2 import imread, remap, imwrite, matchTemplate, circle, line, putText, resize, imshow, waitKey, INTER_CUBIC, IMREAD_GRAYSCALE, TM_CCOEFF_NORMED, FONT_HERSHEY_DUPLEX
from math import sqrt, atan2, pi
from scipy.optimize import curve_fit
from glob import glob
from os.path import split, join

class fisheye():
    def __init__(self):
        
        # images
        self.chessboard_image = None
        self.chessboard_image_gray = None
        self.lcorner_image = None
        self.rcorner_image = None
        
        # points for correction
        self.chessboard_points = None
        self.corner_L = None
        self.corner_L = None
                
        # arrays for re-mapping
        self.map_x = None
        self.map_y = None
        self.radial_distortion_params = None
        pass
    
    def LoadCorrectionMap(self, filename):
        tmp = np.load(filename)
        self.map_x = tmp[0]
        self.map_y = tmp[1]
        
    def SaveCorrectionMap(self, filename):
        tmp = [self.map_x, self.map_y]
        np.save(filename, tmp)

    
    def CorrectPictures(self, filename_mask, destination_folder):
        img_names = glob(filename_mask)
        
        for fn in img_names:
            print 'processing %s...\n' % fn,
            
            orig = imread(fn)
            corrected = remap(orig, f.map_x, f.map_y, INTER_CUBIC)
            imwrite(join(destination_folder, split(fn)[-1]), corrected)  
        
        pass
    
    def fit_func(self, x, a1, a2):
        return a1*np.arctan(x*a2)

    def remap_rad(self, r):
        return self.fit_func(r, self.radial_distortion_params[0], self.radial_distortion_params[1])
        
    def LoadChessboardPicture(self, chessboard_filename, lcorner_filename, rcorner_filename):
        self.chessboard_image_gray = imread(chessboard_filename, IMREAD_GRAYSCALE)
        self.chessboard_image      = imread(chessboard_filename)
        self.lcorner_image = imread(lcorner_filename,IMREAD_GRAYSCALE)
        self.rcorner_image = imread(rcorner_filename,IMREAD_GRAYSCALE)
    
    def FindChessboardRefPoints(self, threshold = 0.70):
        w, h = self.lcorner_image.shape[::-1]

        # find chessboard corners by matching templates (for left and right corners)
        res = matchTemplate(self.chessboard_image_gray, self.lcorner_image,TM_CCOEFF_NORMED)
        loc_L = np.where( res >= threshold)

        res = matchTemplate(self.chessboard_image_gray,self.rcorner_image,TM_CCOEFF_NORMED)
        loc_R = np.where( res >= threshold)

        # merge together points that are too close (for left corners)
        self.corner_L = []
        for pt in zip(*loc_L[::-1]):  
            append = True
            for i in range(len(self.corner_L)):
        
                pt1,n = self.corner_L[i]
        
                # if pt1 and pt are close enough, merge the points
                if (((pt[0]-pt1[0])**2+(pt[1]-pt1[1])**2)<(w**2)):
                    append = False
                    self.corner_L[i] = (((pt1[0]*n + pt[0]+w/2.0)/(n+1.0), (pt1[1]*n + pt[1]+h/2.0)/(n+1.0)), n+1.0)
            
                # else, append pt
            if append: self.corner_L.append(((pt[0]+w/2.0,pt[1]+h/2.0),1.0))

        # merge together points that are too close (for right corners)
        self.corner_R = []
        for pt in zip(*loc_R[::-1]):  
            append = True
            for i in range(len(self.corner_R)):
        
                pt1,n = self.corner_R[i]
        
                # if pt1 and pt are close enough, merge the points
                if (((pt[0]-pt1[0])**2+(pt[1]-pt1[1])**2)<(w**2)):
                    append = False
                    self.corner_R[i] = (((pt1[0]*n + pt[0]+w/2.0)/(n+1.0), (pt1[1]*n + pt[1]+h/2.0)/(n+1.0)), n+1.0)
            
                # else, append pt
            if append: self.corner_R.append(((pt[0]+w/2.0,pt[1]+h/2.0),1.0))

        # keep only points which were merged several times (they must be true...) 
        nn = np.mean([x for p,x in self.corner_L]+[x for p,x in self.corner_R]) / 4.0
        self.data_points = [(pt[0], pt[1], 'L', None, None) for pt,n in self.corner_L if n > nn]
        self.data_points.extend([(pt[0], pt[1], 'R', None, None) for pt,n in self.corner_R if n > nn])

    def find_central_point(self, points):
        '''
        find the closest point to the center of the picture (just to start indexing with (0,0)
        '''
        id_min = 0
        r_min = 1e10
        xm = 0
        ym = 0
        for p in points:
            xm += p[0]
            ym += p[1]
            
        xm /= float(len(points))
        ym /= float(len(points))
        
        #print xm
        #print ym
        for i in range(len(points)):
            p = points[i]
            r = sqrt((p[0]-xm)**2 + (p[1]-ym)**2)
            if (r<r_min):
                r_min = r
                id_min = i
                
        return(id_min)

    def joinbyline(self, points, start_point, id_list, index_inc = (1,0), direction = 0, angle_tol = 30.0, exp_len = -1):
        '''
        points:        list of points
        start_point:   index of the starting point
        index_inc:     increment for the indexing
        direction:     direction to search for next points, in degrees
        angle_tol:     tolerance for the direction to search for next points
        exp_len:       expected length
        '''
        (x, y, s, idx, idy) = points[start_point]
        r = 1e10
        np = -1
        dp = 0
        
        for i in range(len(points)):
            if i == start_point: continue # skip starting point from the search
            
            (xp, yp, sp, idxp, idyp) = points[i]
            
            if sp == s: continue # skip is points belong to the same type (L or R)
            
            da = atan2(yp-y, xp-x)*180.0/pi - direction
            rp = sqrt((xp-x)**2+(yp-y)**2)
            
            if da>180.0:  da -= 360.0
            if da<-180.0: da += 360.0
            
            if (da>angle_tol) or (da<-angle_tol): continue # skip points too far from direction
            
            if (exp_len > 0) and ((rp/exp_len > 1.5) or (rp/exp_len < 0.5)): continue # skip if length check is disabled (-1) or is current length is above +/=20% of expected
            
            if (rp<r):
                r = rp
                np = i
                dp = direction+da
                
        if np != -1: # found another point...
            #print np
            if (points[np][3]==None) and (points[np][4]==None):
                id_list.append(np)
                points[np] = (points[np][0], points[np][1], points[np][2], points[start_point][3]+index_inc[0], points[start_point][4]+index_inc[1])
                self.joinbyline(points, np, id_list, index_inc, dp, angle_tol,r)
        
    def FindChessboard(self):
        '''
        This function organizes in a chessboard grid the sparse points found with the matching template function
        '''
        
        # start with point (0,0): this is the closest point to the center of the picture
        i = self.find_central_point(self.data_points)
        self.data_points[i] = (self.data_points[i][0], self.data_points[i][1], self.data_points[i][2], 0, 0)
        
        id_list = [i]
        
        # other chessboard points are found moving through lines at 0, 90, 180 and 270 degrees
        self.joinbyline(self.data_points, i, id_list, index_inc = (1,0), direction = 0)
        self.joinbyline(self.data_points, i, id_list, index_inc = (-1,0), direction = 180)
        
        for i in id_list:
            self.joinbyline(self.data_points, i, [], index_inc = (0,1), direction = 90)
            self.joinbyline(self.data_points, i, [], index_inc = (0,-1), direction = -90)
    
    
    def PlotChessboard(self):
        for ii in range(len(self.data_points)-1):
            p1 = self.data_points[ii]
            for jj in range(ii+1, len(self.data_points)):
                p2 = self.data_points[jj]
                if (p1[3]!=None) and (p2[3]!=None) and (p1[4]!=None) and (p2[4]!=None) and ((abs(p1[3]-p2[3]) + abs(p1[4]-p2[4])) > 0.99) and ((abs(p1[3]-p2[3]) + abs(p1[4]-p2[4])) < 1.01):
                    line(self.chessboard_image, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))),[0,255,255],5)

        for pt1 in self.data_points:
            circle(self.chessboard_image, (int(round(pt1[0])), int(round(pt1[1]))), 15,[255,0,0],5)

        #for pt1,n in self.corner_R:
        #    if n>10: circle(self.chessboard_image, (int(round(pt1[0])), int(round(pt1[1]))), 15,[0,0,255],5)
            
        for p in self.data_points:
            if (p[3]!=None) and (p[4]!=None):
                putText(self.chessboard_image, '(%.1f,%.1f)' % (p[3], p[4]), (int(round(p[0])), int(round(p[1]))), FONT_HERSHEY_DUPLEX, 1, [0,0,255],3)

        small = resize(self.chessboard_image, (0,0), fx=0.35, fy=0.35) 
        imshow('Chessboard', small ); 
        waitKey() 

    def InterpolateChessboardIndexes(self):
        
        w, h = self.chessboard_image_gray.shape[::-1]
        
        p00  = [x for x in self.data_points if x[3]==0 and x[4]==0]
        p10  = [x for x in self.data_points if x[3]==1 and x[4]==0]
        pm10 = [x for x in self.data_points if x[3]==-1 and x[4]==0]
        p01  = [x for x in self.data_points if x[3]==0 and x[4]==1]
        p0m1 = [x for x in self.data_points if x[3]==0 and x[4]==-1]
        
        xoff = np.interp(w/2.0, [pm10[0][0], p00[0][0], p10[0][0]], [-1,0,1])
        yoff = np.interp(h/2.0, [p0m1[0][1], p00[0][1], p01[0][1]], [-1,0,1])
        
        for i in range(len(self.data_points)):
            if self.data_points[i][3] != None: self.data_points[i] = (self.data_points[i][0], self.data_points[i][1], self.data_points[i][2], self.data_points[i][3]-xoff, self.data_points[i][4])
            if self.data_points[i][4] != None: self.data_points[i] = (self.data_points[i][0], self.data_points[i][1], self.data_points[i][2], self.data_points[i][3], self.data_points[i][4]-yoff)
    
    def ComputeCorrectionMap(self):

        img = self.chessboard_image_gray
        rows,cols = img.shape[:2]
        
        self.map_x = np.zeros(img.shape[:2],np.float32)
        self.map_y = np.zeros(img.shape[:2],np.float32)
        
        
        for j in xrange(rows):
            #print j
            for i in xrange(cols):
                
                x_dest = i - cols/2.0
                y_dest = j - rows/2.0
                
                r_dest = sqrt(x_dest**2 + y_dest**2)
                
                r_orig = self.remap_rad(r_dest)
                
                if r_dest==0:
                    x_orig = x_dest
                    y_orig = y_dest
                else:
                    x_orig = x_dest*r_orig/r_dest + cols/2.0
                    y_orig = y_dest*r_orig/r_dest + rows/2.0

                self.map_x.itemset((j,i),x_orig)
                self.map_y.itemset((j,i),y_orig)
        

    def Estimate_radial_distortion(self, zoom = 1.0):
        good_points = [x for x in self.data_points if (x[3] != None) and (x[4] != None)]
        w, h = self.chessboard_image_gray.shape[::-1]
        
        r_orig = [np.sqrt((x[0]-w/2.0)**2+(x[1]-h/2.0)**2) for x in good_points]
        r_corr = [np.sqrt((x[3])**2+(x[4])**2) for x in good_points]
        
        params = curve_fit(self.fit_func, r_corr, r_orig, [1, 1])
        
        [a1,a2]=params[0]
        
        #r = [self.fit_func(x,a1,a2) for x in r_corr]
        #plt.plot(r_corr,r_orig,'o')
        #plt.plot(r_corr,r,'o')
        #plt.show()        
        
        # normalize polynomials coefficients (scale r_corr to fit picture scale and not chessboard scale)
        a = np.max(r_corr)
        r = a
        target = h/2.0

        
        for i in range(20):
            a /= 2.0
            r1 = self.fit_func(r,a1,a2)
            if r1 > target/zoom: r -=a
            else: r +=a
            
        k = r/target
        
        a2 *= k
        
        self.radial_distortion_params= [a1, a2]
        print params[0], self.radial_distortion_params   
              

                
      

        
    
    
    
# ---------------------------------------------------------------------------- #    
if __name__ == '__main__':
    
    f = fisheye()
    #calib = True
    calib = False
    
    if calib:
        print 'Loading reference chessboard picture and markers...'
        f.LoadChessboardPicture(r'original_pictures\chessboard.JPG', r'markers\marker.png', r'markers\marker_90.png')

        print 'Finding chessboard...'
        f.FindChessboardRefPoints()
        f.FindChessboard()        
        f.InterpolateChessboardIndexes()
        f.PlotChessboard()
        
        print 'Estrapolate correction coefficients... '        
        f.Estimate_radial_distortion(1.0)
        
        print 'Computing correction map...'
        f.ComputeCorrectionMap()
        
        print 'Saving remap matrix on file...'
        f.SaveCorrectionMap('corr')

        print 'Saving example'
        orig = imread(r"original_pictures\chessboard.JPG")
        warped = remap(orig, f.map_x, f.map_y, INTER_CUBIC)
        imwrite(r"corrected_pictures\chessboard.JPG", warped)           
        
        print 'All done!!!'
        
        #waitKey()        
    else:
        f.LoadCorrectionMap('corr.npy.')
        f.CorrectPictures(r"original_pictures\*.JPG", r"corrected_pictures")
        