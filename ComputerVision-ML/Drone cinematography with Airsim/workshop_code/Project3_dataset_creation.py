import setup_path
import airsim
import threading
import json
from PIL import Image
import time
import random

import numpy as np
import os
import tempfile
import pprint
import cv2


image_index = 0


 # Create a dictionary that maps from segmentation rgb values to object ids based on the seg_rgbs.txt file
def get_seg_dictionary():

    f = open('seg_rgbs.txt', 'r')

    d = {}
    
    # We need this dict in order to map from labels to incremental integers (essential for deep learning)
    label_to_incremental = {28:0, 72:1, 95:2, 148:3, 170:4, 241:5}
    
    # 
    def add_spaces_infront(a):
        while len(a) < 3:
            a = ' ' + a
        return a
    
    for line in f.readlines():
        line = line.split('\t')
        
        rgb = json.loads(line[1])[::-1]
        rgb = [add_spaces_infront(str(i)) for i in rgb]
        rgb = str(rgb).replace(',','').replace('\'','')
        
        d[rgb] = int(line[0]) if int(line[0]) not in label_to_incremental.keys() else label_to_incremental[int(line[0])]
    return d



def continuous_shooting(stop):
    global image_index
    
    # create a dictionary that maps from segmentation rgb values to object ids.
    # We need to map RGB colors (that's the output segmentation map format followed by airsim) to incremental object ids
    # (so the final output will be a 2D array with the same spatial dimensions with the input image and object ids as values)
    # in order to comply with the deep learning techniques.
    color2index = get_seg_dictionary()
    
    
    # wait till the UAV is above the ground level to start taking pictures
    while True:
        if image_client.simGetVehiclePose().position.z_val < 0:
            break
            
    # As soon as the UAV is positioned above the ground, start taking pictures.
    while True:
        # get camera images from the UAV
        responses = image_client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
        print('Retrieved images: %d' % len(responses))

        tmp_dir = './segmentation_dataset'
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        for idx, response in enumerate(responses):
            filename = os.path.join(tmp_dir, ['input_imgs/scene', 'target_imgs/segmentation'][idx] + '_' + str(image_index))
            
            if idx==1: # this condition is destined only for segmentation images.
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                if img1d.size == 0: continue
                img_rgb = img1d.reshape(response.height, response.width, 3)
                
                # this command does the conversion from color segmentation map values to integer ids.
                seg = np.apply_along_axis(lambda rgb: color2index[str(rgb)], 2, img_rgb)
                                
                seg = Image.fromarray(seg)#, mode='L') 
                seg.save(os.path.normpath(filename + '.png')) # save the segmentation image
            else: 
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) # save the scene rgb image
        image_index += 1
        
        # Kill signal, otherwise it would continue to take images indefinitely.
        if stop():
            print("Exiting image capturing loop.")
            break



if __name__ == '__main__':
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # create a 2nd identical airsim client in order to not "overload" the first one with image requests.
    image_client = airsim.MultirotorClient()
    image_client.confirmConnection()
    image_client.enableApiControl(True)


    #airsim.wait_key('Press any key to takeoff')
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()

    # Create a thread in order to take images while moving. Otherwise, the pictures would be taken 
    # after reaching the destination point in space (Due to the .join() command).
    stop_thread = False
    x = threading.Thread(target=continuous_shooting, args=(lambda: stop_thread,))
    x.start()
    
    # Move to 80 random points inside a predefined cube. This cube is hard-coded usind two extreme points p1 and p2.
    for i in range(80):
        p1 = (-2.5, -98.7, -38.6)
        p2 = (90, 63.4, -1)
        x = random.uniform(p1[0], p2[0])
        y = random.uniform(p1[1], p2[1])
        z = random.uniform(p1[2], p2[2])
        client.moveToPositionAsync(x, y, z, 5, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0)).join() #, drivetrain=airsim.DrivetrainType.ForwardOnly
    
    # After execution of all random paths, kill the dataset collection thread.
    stop_thread = True

    airsim.wait_key('Press any key to reset to original state')

    client.reset()
    client.armDisarm(False)
    client.enableApiControl(False)
