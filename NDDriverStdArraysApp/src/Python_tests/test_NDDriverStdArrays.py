import unittest
import epics
import numpy as np
import time

P = '13NDSA1:'
R = 'cam1:'
I = 'image1:'

data_types = {
    'int8'   : 0,
    'uint8'  : 1,
    'int16'  : 2,
    'uint16' : 3,
    'int32'  : 4,
    'uint32' : 5,
    'float32': 6,
    'float64': 7
}

np_data_types = {
    'int8'   : np.int8,
    'uint8'  : np.uint8,
    'int16'  : np.int16,
    'uint16' : np.uint16,
    'int32'  : np.int32,
    'uint32' : np.uint32,
    'float32': np.float32,
    'float64': np.float64
}


def dist(nx=256,ny=None):
    ''' Implements the IDL dist() function in Python
	'''
    if ny is None:
        ny = nx
    x = np.linspace(start=-nx/2,stop=nx/2-1, num=nx)
    y = np.linspace(start=-ny/2,stop=ny/2-1, num=ny)
    xx = x + 1j * y[:,np.newaxis]
    out = np.roll(np.roll(np.abs(xx),nx/2,1),ny/2,0)
    return out

def shift(array=None, sx=0, sy=0, sz=0):
    ''' Implements the IDL shift() function in Python
    '''
    if array is None:
        return array
    ndim = len(array.shape)
    if ndim > 3:
        print 'Maximum number of dimensions for array is 3'
        return array
    s = [sx, sy, sz]
    for i in range(ndim):
        array = np.roll(array,s[i],ndim-i-1)
    return array

class TestADSoft(unittest.TestCase):
    
    def setUp(self):
        print('Begin setup')
        self.acquire_pv = epics.PV(P+R+'Acquire.VAL')
        self.image_plugin_array_pv = epics.PV(P+I+'ArrayData.VAL')
        self.array_in_pv = epics.PV(P+R+'ArrayIn')
        self.callback_mode_pv = epics.PV(P+I+'CallbackMode.VAL')
        self.ndimensions_pv = epics.PV(P+R+'NDimensions.VAL')
        self.dimensions_pv = epics.PV(P+R+'Dimensions.VAL')
        self.data_type_pv = epics.PV(P+R+'DataType.VAL')
        self.color_mode_pv = epics.PV(P+R+'ColorMode.VAL')
        self.num_images_pv = epics.PV(P+R+'NumImages.VAL')
        self.append_mode_pv = epics.PV(P+R+'AppendMode.VAL')
        self.new_array_pv = epics.PV(P+R+'NewArray.VAL')
        self.array_complete_pv = epics.PV(P+R+'ArrayComplete.VAL')
        self.num_elements_pv = epics.PV(P+R+'NumElements_RBV.VAL')
        self.image_mode_pv = epics.PV(P+R+'ImageMode.VAL')
        self.detector_state_pv = epics.PV(P+R+'DetectorState_RBV.VAL')
        self.array_callbacks_pv = epics.PV(P+R+'ArrayCallbacks.VAL')
        print('PVs connected')

        self.array_callbacks_pv.put('Enable')
        print('Completed setup')

    def tearDown(self):
        pass

    def test_append_disable(self):
        # Test AppendMode=Disable
        self.append_mode_pv.put('Disable')
        self.image_mode_pv.put('Single')
        nx, ny = 256, 768
        result = True
        scale = 100
        scaleStep = 1.05
        self.ndimensions_pv.put(2)
        self.dimensions_pv.put([nx, ny, 0, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('Mono')
        for dt in data_types:
            print('Testing append mode disable {:s}'.format(dt))
            self.data_type_pv.put(data_types[dt])
            row = np.arange(nx) / (nx-1.) * scale
            scale *= scaleStep
            image = np.tile(row, ny).reshape(ny, nx).transpose()
            image = image.astype(np_data_types[dt])
            self.acquire_pv.put('Acquire')
            self.array_in_pv.put(image.flatten(order='F'), wait=True)
            time.sleep(.5)
            rimage = self.image_plugin_array_pv.get(count=nx*ny).reshape(nx,ny,order='F')
            result = result and self.assertTrue(np.array_equal(image, rimage))
        return result

    def test_append_enable(self):
        # Test AppendMode=Enable 
        self.append_mode_pv.put('Enable')
        self.image_mode_pv.put('Single')
        nx, ny = 256, 768
        result = True
        ystep = 1
        scale = 50
        scaleStep = 1.1
        self.ndimensions_pv.put(2)
        self.dimensions_pv.put([nx, ny, 0, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('Mono')
        for dt in data_types:
            print('Testing append mode enable {:s}'.format(dt))
            ystart = 0
            self.new_array_pv.put(1)
            self.data_type_pv.put(data_types[dt])
            self.acquire_pv.put('Acquire')
            row = np.arange(nx) / (nx-1.) * scale
            scale *= scaleStep
            image = np.tile(row, ny).reshape(ny, nx).transpose()
            image = image.astype(np_data_types[dt])
            while ystart < ny:
              chunk = image[:, ystart:ystart+ystep].flatten(order='F')
              self.array_in_pv.put(chunk, wait=True)
              ystart = ystart + ystep
              #time.sleep(0.001)
            time.sleep(0.5)
            self.array_complete_pv.put(1)
            rimage = self.image_plugin_array_pv.get(count=nx*ny).reshape(nx,ny,order='F')
            result = result and self.assertTrue(np.array_equal(image, rimage))
        return result

    def test_mono(self):
        # Test Mono sine waves
        self.append_mode_pv.put('Disable')
        self.image_mode_pv.put('Continuous')
        result = True
        nx, ny = 500,800
        scale = 100
        dt = 'float32'
        self.ndimensions_pv.put(2)
        self.dimensions_pv.put([nx, ny, 0, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('Mono')
        self.data_type_pv.put(data_types[dt])
        d = dist(nx, ny)
        for i in range(1, 50):
            print('Testing mono mode ', i)
            image = scale*np.sin(shift(d/i, nx/2, ny/2))
            image = image.astype(np_data_types[dt]).transpose()
            self.acquire_pv.put('Acquire')
            self.array_in_pv.put(image.flatten(order='F'), wait=True)
            rimage = self.image_plugin_array_pv.get(count=nx*ny).reshape(nx,ny,order='F')
            result = result and self.assertTrue(np.array_equal(image, rimage))
        return result

    def test_rgb1(self):
        # Test rgb1 sine waves
        self.append_mode_pv.put('Disable')
        self.image_mode_pv.put('Continuous')
        result = True
        nx, ny = 600,800
        scale = 100
        dt = 'int8'
        self.ndimensions_pv.put(3)
        self.dimensions_pv.put([3, nx, ny, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('RGB1')
        self.data_type_pv.put(data_types[dt])
        d = dist(nx, ny)
        image = np.empty([3, nx, ny])
        for i in range(1, 20):
            print('Testing RGB1 mode ', i)
            red   = scale*np.sin(shift(d/i,     nx/2, ny/2))
            green = scale*np.sin(shift(d/(i*2), nx/2, ny/2))
            blue  = scale*np.sin(shift(d/(i*3), nx/2, ny/2))
            image[0,:,:] = red.transpose()
            image[1,:,:] = green.transpose()
            image[2,:,:] = blue.transpose()
            image = image.astype(np_data_types[dt])
            self.acquire_pv.put('Acquire')
            self.array_in_pv.put(image.flatten(order='F'), wait=True)
            rimage = self.image_plugin_array_pv.get(count=3*nx*ny).reshape(3,nx,ny,order='F')
            result = result and self.assertTrue(np.array_equal(image, rimage))
        return result

    def test_scan_sim1(self):
        # Test scan record simulation 
        self.append_mode_pv.put('Enable')
        self.image_mode_pv.put('Single')
        ndet, nx, ny = 4, 100, 200
        result = True
        self.ndimensions_pv.put(3)
        self.dimensions_pv.put([ndet, nx, ny, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('Mono')
        print('Testing scan simulation mode')
        self.acquire_pv.put('Acquire')
        self.new_array_pv.put(1)
        dt = 'float64'
        self.data_type_pv.put(data_types[dt])
        counts = np.empty(ndet)
        scale = 100.
        period = 10.
        for y in range(ny):
            for x in range(nx):
                counts[0] = x + y;
                counts[1] = (x + y) * np.random.random();
                counts[2] = x*np.random.random();
                counts[3] = scale * np.sin(((x-nx/2.)**2 + (y-ny/2.)**2) / period)
                self.array_in_pv.put(counts, wait=True)
        self.array_complete_pv.put(1)
        return result

if __name__ == '__main__':
    unittest.main()
