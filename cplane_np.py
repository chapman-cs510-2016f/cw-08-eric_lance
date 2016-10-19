#!/usr/bin/env python3

import numpy as np
import pandas as pd
import csv                # for I/O with csv format files
import json               # for I/O with json formatted data
import abscplane as absc
import matplotlib.pyplot as plt
import numba as nb        # Just-In-Time compilation

class ComplexPlaneNP(absc.AbsComplexPlane):
    """This is the Class ComplexPlaneNP.  It is built from the Abstract Class AbsComplexPlane.
    This Class serves as a simplistic pan/zoom over a 2D complex plane, where each point in the
    plane undergoes a transformation through the function f(), where the value at the coordinate
    point is:
        value = f( x + yj )

    This class uses the data structures available in numpy and pandas.  The array data structure
    is used from numpy to set up the rows for the plane.  The rows are then inserted into a pandas
    DataFrame and given row and column names to help identify the rows and columns in the plane.

    The contents of each 'cell' in the ComplexPlaneNP is of type imaginary number.
    """

    def __init__(self, newXmin=-5., newXmax=5., newXlen=1001, newYmin=-5., newYmax=5., newYlen=1001, f=lambda x: x, maxLoop=100):
        """This is the creator.  It can be passed the the min/max X and Y values for the plane,
        and a transformation function (f).  There are default values if the parameters are not
        passed to the creator.
        The creator then generates a 2D plane filled with the X & Y complex number coordinates
        of the specified plane transformed by the function f().  Note that the default function
        f() for computing the values in the plane is the identity function, so the values at
        the coordinate location are the coordinates themselves.  Note also that the number of
        points in each axis is always forced to be fixed value.
        """
        self.xmin = newXmin
        self.xmax = newXmax
        self.ymin = newYmin
        self.ymax = newYmax
        self.xlen = newXlen
        self.ylen = newYlen
        #  must sub 1 to get the correct actual step size, otherwise last element does not equal x or y max
        self.xstep = (self.xmax - self.xmin)/(self.xlen - 1)
        self.ystep = (self.ymax - self.ymin)/(self.ylen - 1)
        self.f = f
        self.max = maxLoop
        #  call refresh() to generate the the plane and its contents
        self.refresh()



    def refresh(self):
        """Regenerate complex plane.
        For every point (x + y*1j) in self.plane, replace
        the point with the value self.f(x + y*1j). 
        """
        rx = np.linspace( self.xmin, self.xmax, self.xlen )
        ry = np.linspace( self.ymin, self.ymax, self.ylen )
        x, y = np.meshgrid( rx, ry )
        planeArray = x + y*1j
        self.f( planeArray )
        ylabels = [str(self.ymax-ypos*self.ystep) for ypos in range(self.ylen)]
        xlabels = [str(xpos*self.xstep+self.xmin) for xpos in range(self.xlen)]
        self.plane = pd.DataFrame(planeArray, index=ylabels, columns=xlabels)



    def zoom(self,newXmin,newXmax,newYmin,newYmax):
        """Reset self.xmin, self.xmax, and/or self.xlen.
        Also reset self.ymin, self.ymax, and/or self.ylen.
        Zoom into the indicated range of the x- and y-axes.
        Refresh the plane as needed."""
        # note that xstep and ystep must be recalculated for the new min and max
        self.xmin = newXmin
        self.xmax = newXmax
        self.ymin = newYmin
        self.ymax = newYmax
        self.xstep = (self.xmax - self.xmin)/(self.xlen - 1)
        self.ystep = (self.ymax - self.ymin)/(self.ylen - 1)
        self.refresh()



    def set_f(self, f):
        """Reset the transformation function f.
        Refresh the plane as needed."""
        self.f = f
        self.refresh()


class JuliaPlane(ComplexPlaneNP):
    """This is the Class JuliaPlane.  It is built from the Class ComplexPlaneNP.
    This Class serves as a simplistic pan/zoom over a 2D complex plane, where each point in the
    plane undergoes a transformation through a function f() created by a call to the function julia().
    This transformation function returns an integer, which is stored in the JuliaPlane.
    NOTE that the julia function is included in this source file, but is not part of the JuliaPlane()
    class.

    The contents of each 'cell' in the JuliaPlane is of type integer.
    """

    def __init__(self, newXmin=-5., newXmax=5., newXlen=11, newYmin=-5., newYmax=5., newYlen=11, c=(-1.037 + 0.17j), maxLoop=100):
        """ The JuliaPlane creator method uses the ComplexPlaneNP creator to generate the initial 2D plane.
        The function for this plane is then reset to a new function, and the values re-generated.
        Note that since this function was intially created, the f parameter was added to ComplexPlaneNP's
        creator, but this code was not updated to take advantage of the passed parameter change.
        """
        #  set the function and re-compute the plane's values
        f = julia(c, maxLoop)
        self.c = c
        ComplexPlaneNP.__init__(self, newXmin, newXmax, newXlen, newYmin, newYmax, newYlen, f, maxLoop)

    def show(self, chosenmap=plt.cm.hot):
        """This method plots an image of the contents of the 2D complex plane.  The numbers in the plane
        are treated as gray-scale in matplotlib.imshow(), with an optional color map being used to turn
        the gray-scale into various color combinations.
        """
        plt.clf()
        plt.imshow(self.plane.as_matrix(), cmap=chosenmap, interpolation='bicubic', extent=(self.xmin, self.xmax, self.ymin, self.ymax))
        plt.title( 'c = '+str(self.c) )
        plt.show()

    def set_f(self, c, max=100):
        """This method is used to set the transformation function in the ComplexPlane for this JuliaPlane.
        The function julia is currently not a member of JuliaPlane.
        This method sets the complex value 'c' in the julia function and defaults to a maximum iteration count
        of 100.
        """
        self.c = c  # keep a copy for the CSV and JSON output
        self.f = julia(c, max)
        self.refresh()


    def toCSV( self, filename ):
        """Output the contents of the julia plane and all parameters needed to recreate it in CSV format
        Created with the help of https://docs.python.org/3/library/csv.html
        """
        #  open the file for writing
        with open(filename, 'w', newline='' ) as csvfile:
            writer = csv.writer( csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL )

            # write out the parameters needed to rebuild the julia plane
            writer.writerow( [ 'JuliaPlane', 'xmin', 'xmax', 'xlen', 'ymin', 'ymax', 'ylen', 'c' ] )
            writer.writerow( [ '', self.xmin, self.xmax, self.xlen, self.ymin, self.ymax, self.ylen, self.c ] )
            # blank row
            writer.writerow( [ 'JuliaPlane Contents' ] )
            #  write out the contents of the plane
            for row_i in self.plane.iterrows():
                writer.writerows( row_i[1:self.xlen] )

            #  we're done, clean up the file
            csvfile.close()


    def fromCSV( self, filename ):
        """Read in the contents of a csv file to rebuild a save julia plane.  We are only interested in
        the parameters necessary to reconstruct the plane, we do not need to read the contents of the plane
        itself"""
        #  open the file for reading
        with open( filename, newline='' ) as csvfile:
            reader = csv.reader( csvfile, delimiter=',', quotechar='"' )
            count = 1
            for row in reader:
                if count == 2:
                    #  parse the second row only
                    self.xmin  = float( row[ 1 ] )
                    self.xmax  = float( row[ 2 ] )
                    self.xlen  = int( row[ 3 ] )
                    self.ymin  = float( row[ 4 ] )
                    self.ymax  = float( row[ 5 ] )
                    self.ylen  = int( row[ 6 ] )
                    self.c     = complex( row[ 7 ] )
                    self.xstep = (self.xmax - self.xmin)/(self.xlen - 1)
                    self.ystep = (self.ymax - self.ymin)/(self.ylen - 1)
                    self.set_f( self.c )      # set the tranformation function.  Note that this automatically calls refresh()
                    break
                elif count == 1:
                    #  this is a header row, which we can skip
                    count += 1
                else:
                    # discard all subsequent rows
                    break

            #  we're done, clean up the file
            csvfile.close()


    def toJSON( self, filename ):
        """Output the contents of the julia plane and all parameters needed to recreate it in JSON encoded format
        Created with the help of http://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file-in-python
        """
        #  open the file for writing
        with open( filename, 'w') as jsonfile:
            #  output the parameters needed to recreate the plane
            data = { "JuliaPlaneParameters": { "xmin":self.xmin, "xmax":self.xmax, "xlen":self.xlen, "ymin":self.ymin, "ymax":self.ymax, "ylen":self.ylen, "creal":self.c.real, "cimaginary":self.c.imag }}

            #  handle the output of the plane contents
            mat = self.plane.as_matrix()
            for row in range( 0, self.xlen ):
                print( mat[ row ] )
                l = mat[ row ].tolist()
                data[ "JuliaPlaneContents" + str( row )] = { self.plane.index[ row ]:  l }

            #  write the accummulated dictionary to the JSON file
            json.dump(data, jsonfile, indent=4, sort_keys=True, separators=(',', ':'))

            #  we're done, clean up the file
            jsonfile.close()


    def fromJSON( self, filename ):
        """Read in the contents of a JSON file to rebuild a save julia plane.  We are only interested in
        the parameters necessary to reconstruct the plane, we do not need to read the contents of the plane
        itself"""
        #  open the file for reading
        from pprint import pprint

        #  open and read in the json file, with help from:  http://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file-in-python
        with open( filename ) as jsonfile:    
            data = json.load(jsonfile)

            #  parse the parameter values
            self.xmin = data["JuliaPlaneParameters"]["xmin"]
            self.xmax = data["JuliaPlaneParameters"]["xmax"]
            self.xlen = int( data["JuliaPlaneParameters"]["xlen"] )
            self.ymin = data["JuliaPlaneParameters"]["ymin"]
            self.ymax = data["JuliaPlaneParameters"]["ymax"]
            self.ylen = int( data["JuliaPlaneParameters"]["ylen"] )
            self.c    = data["JuliaPlaneParameters"]["creal"] + data["JuliaPlaneParameters"]["cimaginary"]*1j
            self.xstep = (self.xmax - self.xmin)/(self.xlen - 1)
            self.ystep = (self.ymax - self.ymin)/(self.ylen - 1)
            self.set_f( self.c )      # set the tranformation function.  Note that this automatically calls refresh()

            #  uncomment this line to see the re-constituted plane
#            print( self.plane )

            #  we're done, clean up the file
            jsonfile.close()

class JuliaPlaneNV(JuliaPlane):
    """This is the Class JuliaPlaneNV.  It is built from the Class JuliaPlane, but is not vectorized.
    This Class serves as a simplistic pan/zoom over a 2D complex plane, where each point in the
    plane undergoes a transformation through a function f() created by a call to the function julia().
    This transformation function returns an integer, which is stored in the JuliaPlane.
    NOTE that the julia function is included in this source file, but is not part of the JuliaPlaneNV()
    class.

    The contents of each 'cell' in the JuliaPlane is of type integer.
    """
    def __init__(self, newXmin=-5., newXmax=5., newXlen=11, newYmin=-5., newYmax=5., newYlen=11, c=(-1.037 + 0.17j), maxLoop=100):
        """ The JuliaPlane creator method uses the ComplexPlaneNP creator to generate the initial 2D plane.
        The function for this plane is then reset to a new function, and the values re-generated.
        Note that since this function was intially created, the f parameter was added to ComplexPlaneNP's
        creator, but this code was not updated to take advantage of the passed parameter change.
        """
        #  set the function and re-compute the plane's values
        f = juliaNV(c, maxLoop)
        self.c = c
        ComplexPlaneNP.__init__(self, newXmin, newXmax, newXlen, newYmin, newYmax, newYlen, f, maxLoop)

    def refresh(self):
        """Regenerate complex plane.
        For every point (x + y*1j) in self.plane, replace
        the point with the value self.f(x + y*1j). 
        """
        planeArray = np.zeros([self.xlen,self.ylen])
        for xpos in range(self.xlen):
            for ypos in range(self.ylen):
                #  compute the value at each of the coordinate points in the plane
                planeArray[(self.ylen-ypos-1),xpos] = self.f( (xpos*self.xstep+self.xmin) + (ypos*self.ystep+self.ymin)*1j )
        ylabels = [str(self.ymax-ypos*self.ystep) for ypos in range(self.ylen)]
        xlabels = [str(xpos*self.xstep+self.xmin) for xpos in range(self.xlen)]
        self.plane = pd.DataFrame(planeArray, index=ylabels, columns=xlabels)

    def set_f(self, c, max=100):
        """This method is used to set the transformation function in the ComplexPlane for this JuliaPlane.
        The function julia is currently not a member of JuliaPlane.
        This method sets the complex value 'c' in the julia function and defaults to a maximum iteration count
        of 100.
        """
        self.c = c  # keep a copy for the CSV and JSON output
        self.f = juliaNV(c, max)
        self.refresh()


def julia(c, max=100):
    """This method creates and returns a function, f.  The parameters passed to julia are:
    c - an imagery valued constant that is used in the function f.
    max - an optional argument that sets the maximum loop count within f.  Default is 100.

    The function f requires a single parameter:
    z - an imaginary number

    f then performs the operation z = z**2 + c on the z passed to f along with the c value passed to julia.
    The operation is performed up to max times.
    The function f returns:
    1 - if the magnitude of the imagiary number (z) passed in exceeds 2
    n - count of the number of times the operation can be done *before* the magnitude of z exceeds 2
    0 - if the max iterations through the loop is reached without the magnitude of z reaching 2

    Note that f's return value of 1 is ambiguous:  it could be because the initial z was too large,
    or because the operation could be performed once successfully.
    """
    #  this decorator causes this function definition to be JIT
    @nb.vectorize([nb.int32(nb.complex128)])
    def f(z):
        # check to see if the input is already too big
        if abs( z ) <= 2:
            n = 0
            while abs(z)<=2:
                #  perform the operation
                z = z**2 + c
                #  have we exceeded our max loop count-1?
                if n >= max:
                    n = 1
                    break
                #  count the number of times through the loop
                n+=1
            n -= 1  # subtract one to count the total loops *before* exceeding 2, also reports 0 if max loop reached
        else:
            #  report input too big
            n = 1
        return n

    #  return the function pointer to the caller of the julia() method
    return f


def juliaNV(c, max=100):
    """This method creates and returns a function, f.  The parameters passed to julia are:
    c - an imagery valued constant that is used in the function f.
    max - an optional argument that sets the maximum loop count within f.  Default is 100.

    The function f requires a single parameter:
    z - an imaginary number

    f then performs the operation z = z**2 + c on the z passed to f along with the c value passed to julia.
    The operation is performed up to max times.
    The function f returns:
    1 - if the magnitude of the imagiary number (z) passed in exceeds 2
    n - count of the number of times the operation can be done *before* the magnitude of z exceeds 2
    0 - if the max iterations through the loop is reached without the magnitude of z reaching 2

    Note that f's return value of 1 is ambiguous:  it could be because the initial z was too large,
    or because the operation could be performed once successfully.
    """
    def f(z):
        # check to see if the input is already too big
        if abs( z ) <= 2:
            n = 0
            while abs(z)<=2:
                #  perform the operation
                z = z**2 + c
                #  have we exceeded our max loop count-1?
                if n >= max:
                    n = 1
                    break
                #  count the number of times through the loop
                n+=1
            n -= 1  # subtract one to count the total loops *before* exceeding 2, also reports 0 if max loop reached
        else:
            #  report input too big
            n = 1
        return n

    #  return the function pointer to the caller of the julia() method
    return f





#  unit testing functions are in the file test_cplane_np.py

