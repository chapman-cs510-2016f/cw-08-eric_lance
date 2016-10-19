#!/usr/bin/env python3

import numpy as np
import pandas as pd
import cplane_np as jp

"""This file contains the unit test functions for the JuliaPlane class in cplane_np.py
   This file contains the tests for JuliaPlane methods, including I/O, and the base interface functions."""


def _do_julia( c, z, loop_max = 100 ):
    """This private function is for testing julia only"""
    f = jp.julia( c, loop_max )
    return f( z )


def test_julia_1():
    """Test the julia function for returning the correct count"""
    success  = False
    expected = 3

    #  perform the julia test and check the return value
    actual = _do_julia( 0.2 + 0.2j, 0.7 + 0.7j ) 
    if expected == actual:
        success = True

    message = 'Julia function did not return expected count:  actual %d expected %d' % (actual, expected)
    assert success, message


def test_julia_2():
    """Test the julia function for returning the zero if the loop count is exceeded"""
    success  = False
    expected = 0

    #  perform the julia test and check the return value
    actual = _do_julia( 0.1 + 0.1j, 0.1 + 0.1j, 10 )
    if expected ==  actual:
        success = True

    message = 'Julia function was expected to exceed the loop count:  actual %d expected %d' % (actual, expected)
    assert success, message


def test_julia_3():
    """Test the julia function for returning that the magnitude of the starting value is too large"""
    success  = False
    expected = 1

    #  perform the julia test and check the return value
    actual = _do_julia( 2 + 2j, 7 + 7j )
    if expected == actual:
        success = True

    message = 'Julia function did not report the z magnitude already too big:  actual %d expected %d' % (actual, expected)
    assert success, message


def test_julia_4():
    """Test the julia function to make sure a subsequent call to julia does not change the parameters and
    operation of the first call to julia"""
    success  = False

    #  make the first julia function
    f1 = jp.julia( 0.2 + 0.2j, 10 )
    #  make the results from this call the expected outcome
    expected = f1( 0.7 + 0.7j )

    # make the second julia function
    f2 = jp.julia( -0.2 + -0.2j, 2 )

    #  call the first julia function again and make sure the output matches the first call
    actual = f1( 0.7 + 0.7j )

    # how'd we do??
    if expected == actual:
        success = True

    message = 'Julia function internals changed between indpendent instatiations:  actual %d expected %d' % (actual, expected)
    assert success, message


def test_julia_5():
    """Test the julia function to make sure two different function instantiations yield two different results
    even with the same z value passed to the returned functions"""
    success  = False

    #  make the first julia function
    f1 = jp.julia( 0.2 + 0.2j, 10 )

    # make the second julia function
    f2 = jp.julia( -0.2 + -0.2j, 2 )

    #  call the first julia function again and make sure the output matches the first call
    #  make the results from this call the expected outcome
    expected = f1( 0.7 + 0.7j )
    actual = f2( 0.7 + 0.7j )

    # how'd we do??
    if expected != actual:
        success = True

    message = 'Different julia functions constants returned the same value:  f1 %d f2 %d' % (actual, expected)
    assert success, message



def test_init_no_params():
    """Test the creator by passing no parameters.  Since default values are used in place of missing parameters
       this should *not* cause a TypeError exception"""
    success = True

    try:
        testPlane = jp.JuliaPlane()
    except TypeError:
        """test passes"""
        success = False

    message = 'Creator should not have generated a TypeError exception, as all unpassed parameters have default values'
    assert success, message


def test_init():
    """Test the creator by passing the required parameters.  The passed in values should match the object's values"""
    success = True

    try:
        xmin = 2
        xmax = 6
        ymin = -6
        ymax = -2
        xlen = 1001
        ylen = 1001
        testPlane = jp.JuliaPlane( xmin, xmax, ymin, ymax, xlen, ylen )

        #  this line is to force an error to prov the test can fail
        # xmin = xmin + 1

        #  check that the parameters are all correctly stored
        if testPlane.xmin != xmin or testPlane.xmax != xmax or testPlane.ymin != ymin or testPlane.ymax != ymax or testPlane.xlen != xlen or testPlane.ylen != ylen:
           message = 'Init parameter mismatch: expected %d %d %d %d %d %d, actual %d %d %d %d %d %d' % (xmin, xmax, ymin, ymax, xlen, ylen, testPlane.xmin, testPlane.xmax, testPlane.ymin, testPlane.ymax, testPlane.xlen, testPlane.ylen)
           success = False

    except TypeError:
        """Test fails, should not have generated an exception"""
        message = 'Creator generated an exception when correct number of parameters were passed in'
        success = False

    assert success, message


def itest_setf1():
    """Test that setting the function to a new function updates the plane with the new transformation values"""

    #  set the constant c
    c = 0.4 - 0.4j 

    #  create a plane
    tp = jp.JuliaPlane( 0, 10, 0, 10, 21, 21 )
    #  set the function to use c
    tp.set_f( c )

    # set up the expected plane
    xmin = 0
    xmax = 10
    xstep = 0.5
    xlen = 21
    ymin = 0
    ymax = 10
    ystep = 0.5
    ylen = 21
    f = jp.julia( c )
    eplane = [[f(( j*xstep + xmin ) + ( i*ystep + ymin )*1j) for i in range(ylen)] for j in range(xlen)]

    #  this line is to force an error to prove the test can fail
    #eplane[1][1] = 5

    #  do the expected and actual planes match?
    success = tp.plane == eplane
    message = 'set_f() did not correctly transform the plane to double the coordinate values'
    assert success, message



def test_setf2():
    """Set the transformation function to be something other than a function(), which should fail,
    meaning the test was successful"""
    #  create a plane
    tp = jp.JuliaPlane( 1, 10, 1, 10 )

    try:
        #  set the function to be something other than a complex number
        tp.set_f( tp )
        message = 'Test Failed, succeeded in setting function to a non-complex value'
        success = False
    except TypeError:
        """Test succeeds, exception generated"""
        success = True

    assert success, message

def itest_zoom1():
    """Test the zoom function with valid values.  Zoom should move/reset the 2D plane to a known configuration"""
    # set up the expected plane
    xmin = 0
    xmax = 10
    xstep = 0.5
    xlen = 21
    ymin = 0
    ymax = 10
    ystep = 0.5
    ylen = 21
    #  set the constant c
    c = -1.037 + 0.17j
    f = jp.julia( c )
    eplane = [[f(( j*xstep + xmin ) + ( i*ystep + ymin )*1j) for i in range(ylen)] for j in range(xlen)]

    #  create a plane
    tp = jp.JuliaPlane( 100, 200, -100, 0 )
    tp.zoom(xmin, xmax, ymin, ymax)

    #  do the expected and actual planes match?
    success = tp.plane == eplane
    message = 'zoom() did not correctly transform the plane to the new coordinate values'
    assert success, message

def test_zoom2():
    """Test the zoom function with invalid values.  Zoom should generate an exception"""
    #  create a plane
    tp = jp.JuliaPlane( 100, 200, -100, 0 )
    try:
        tp.zoom( "one", 100, -1, 3)
        message = 'Test Failed, zoom did not catch use of an invalid parameter'
        success = False
    except TypeError:
        """Test succeeds, exception generated"""
        success = True

    assert success, message



def itest_refresh1():
    """Test the refresh function.  Create a plane, corrupt the data in the plane, refresh and verify the data is once again correct"""
    #  create a plane
    tp = jp.JuliaPlane( 100, 200, -100, 0 )
    #  create a duplicate plane
    ep = jp.JuliaPlane( 100, 200, -100, 0 )

    #  corrupt the original test plane
    tp.plane = [[(-1) for i in range(tp.ylen)] for j in range(tp.xlen)]

    # refresh the plane
    tp.refresh()

    #  this line is to force an error to prove the test can fail
    #ep.plane[1][1] = -1

    #  do the expected and actual planes match?
    success = tp.plane == ep.plane
    message = 'refresh() did not correctly retore the plane to the expected coordinate values'
    assert success, message

