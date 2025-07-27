import unittest
import numpy as np
import os
scriptDir=os.path.dirname(__file__)
from nalulib.airfoil_shapes import * 
from nalulib.airfoil_shapes_naca import naca_shape

# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
class TestAirfoilShape(unittest.TestCase):

    def test_problematic_airfoils(self):
        # 
        info = []
        info += [('../data/airfoils/tests/naca6412.dat', 'blunt', [60,61, 62, 0])] # Blunt inclined like this: / 
        info += [('../data/airfoils/blunt_not_straight.csv'         , 'blunt' , np.concatenate((np.arange(499,524), [0])))] # Inclined like this: /
        info += [('../data/airfoils/tests/du91-w2-225_nalu_l40.csv' , 'sharp' , np.array([40  , 0]))] # Sharp blunt like this: =>

        #info += [('../data/airfoils/tests/ag04.dat', 'sharp', [90, 0])] # TODO TODO

        #info += [('../data/airfoils/tests/s1221.dat', 'sharp', [90, 0])]
        #info += [('../data/airfoils/tests/goe187.dat', 'sharp', [90, 0])]
        #info += [('../data/airfoils/tests/goe795sm.dat', 'sharp', [90, 0])]

        for fpath, TE_type, ITE in info:
            fullpath = os.path.join(scriptDir, fpath)
            #print(f"------ {fullpath} - TE_type: {TE_type} - ITE: {ITE}")
            arf = StandardizedAirfoilShape(filename =fullpath)
            #arf.plot()
            self.assertEqual(arf._TE_TYPE, TE_type)
            np.testing.assert_equal(arf._ITE, ITE)



    def test_manip_sharp(self):
        # Manipulations for a sharp trailing edge airfoil
        #P=Polar(os.path.join(MyDir,'../data/Cylinder.dat'))
        #self.assertEqual(P.cl.size,3)
        digits='0022'
        n=3
        x, y = naca_shape(digits, chord=1, n=n, sharp=True)
        #print('x',x)
        #print('y',y)

        arf = AirfoilShape(x=x, y=y, name='Naca'+digits)
        self.assertEqual(arf.iTE, 2)
        self.assertEqual(arf.iLE, 0)
        self.assertEqual(arf.closed, True)
        self.assertEqual(arf.orientation, 'clockwise')
        self.assertEqual(arf.startPoint, 'LE')
        self.assertEqual(arf.chord, 1)
        self.assertEqual(len(arf.x), 5)
        np.testing.assert_almost_equal(arf.thickness_max, 0.194114, 3)

        # --- Change orientation (do nothing)
        arf.setOrientation('clockwise')
        self.assertEqual(arf.iTE, 2)
        self.assertEqual(arf.iLE, 0)
        self.assertEqual(arf.orientation, 'clockwise')

        # --- Change orientation
        arf.setOrientation('counterclockwise')
        self.assertEqual(arf.iTE, 2)
        self.assertEqual(arf.iLE, 0)
        self.assertEqual(arf.orientation, 'counterclockwise') #<<<

        # --- Change startPoint (do nothing)
        arf.setStartPoint('LE')
        self.assertEqual(arf.iTE, 2)
        self.assertEqual(arf.iLE, 0)
        self.assertEqual(arf.orientation, 'counterclockwise')

        # --- Change startPoint
        arf.setStartPoint('TE')
        self.assertEqual(arf.iTE, 0) #<<<
        self.assertEqual(arf.iLE, 2) #<<<
        self.assertEqual(arf.orientation, 'counterclockwise')

        arf.setOrientation('clockwise')
        self.assertEqual(arf.iTE, 0)
        self.assertEqual(arf.iLE, 2)
        self.assertEqual(arf.orientation, 'clockwise') #<<<


        # --- Make it closed (do nothing)
        arf.closed=True
        self.assertEqual(arf.iTE, 0)
        self.assertEqual(arf.iLE, 2)
        self.assertEqual(arf.closed, True)
        self.assertEqual(len(arf.x), 5)


        # --- Make it open
        arf.closed=False
        self.assertEqual(arf.iTE, 0)
        self.assertEqual(arf.iLE, 2)
        self.assertEqual(arf.closed, False) #<<<
        self.assertEqual(len(arf.x), 4)     #<<<

        # --- Make it closed
        arf.closed=True
        self.assertEqual(arf.iTE, 0)
        self.assertEqual(arf.iLE, 2)
        self.assertEqual(arf.closed, True) #<<<
        self.assertEqual(len(arf.x), 5)    #<<<

        #arf.write('_Naca{}.csv'.format(digits), format='csv', delim=' ')
        #arf.plot(digits)
        #arf.plot_surfaces()
        #arf.plot()



if __name__ == '__main__':
    #TestAirfoilShape().test_problematic_airfoils()
    unittest.main()
    #import matplotlib.pyplot as plt
    #plt.show()
