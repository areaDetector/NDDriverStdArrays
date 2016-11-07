import unittest
import epics
import numpy as np
import time

nx, ny = 256, 768
P = '13NDSA1:'
R = 'cam1:'
I = 'image1:'

data_types = {
    'int8'  : 0,
    'uint8'   : 1,
    'int16' : 2,
    'uint16'  : 3,
    'int32' : 4,
    'uint32'  : 5,
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

class TestADSoft(unittest.TestCase):
    
    def setUp(self):
        print('Begin setup')
        self.image = np.random.uniform(0, 255, size=nx*ny).astype(np.uint8).reshape(nx, ny)
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

        self.ndimensions_pv.put(2)
        self.dimensions_pv.put([nx, ny, 0, 0, 0, 0, 0, 0, 0, 0])
        self.color_mode_pv.put('Mono')
        self.num_images_pv.put(5)
        self.array_callbacks_pv.put('Enable')
        print('Completed setup')

    def tearDown(self):
        pass

    def test_append_disable(self):
        # Test AppendMode=Disable
        self.append_mode_pv.put('Disable')
        self.image_mode_pv.put('Single')
        result = True
        for dt in data_types:
            print('Testing append mode disable {:s}'.format(dt))
            self.data_type_pv.put(data_types[dt])
            row = np.arange(nx) / (nx-1.) * 100
            image = np.tile(row, (ny,1))
            image = image.astype(np_data_types[dt]).reshape(ny, nx)
            self.acquire_pv.put('Acquire')
            self.array_in_pv.put(image.flatten(), wait=True)
            time.sleep(.1)
            rimage = self.image_plugin_array_pv.get(count=nx*ny).reshape(ny,nx)
            result = result and self.assertTrue(np.array_equal(image, rimage))

        return result

    def test_append_enable(self):
        # Test AppendMode=Enable 
        self.append_mode_pv.put('Enable')
        self.image_mode_pv.put('Single')
        ystart = 0
        ystep = 1
        result = True
        for dt in data_types:
            print('Testing append mode enable {:s}'.format(dt))
            self.new_array_pv.put(1)
            self.data_type_pv.put(data_types[dt])
            self.acquire_pv.put('Acquire')
            row = np.arange(nx) / (nx-1.) * 50
            image = np.tile(row, (ny,1))
            image = image.astype(np_data_types[dt]).reshape(ny, nx)
            while ystart < ny:
              chunk = image[ystart:ystart+ystep,:].astype(np_data_types[dt]).flatten()
              self.array_in_pv.put(chunk, wait=True)
              ystart = ystart + ystep
              time.sleep(0.001)

            time.sleep(0.5)
            self.array_complete_pv.put(1)
            rimage = self.image_plugin_array_pv.get(count=nx*ny).reshape(ny,nx)
            result = result and self.assertTrue(np.array_equal(image, rimage))

        return result

if __name__ == '__main__':
    unittest.main()
