import setup_path
import airsim
import numpy as np


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()





################################################################
#       TAKE A PHOTO OF THE YELLOW BALL
################################################################

pi = 3.14159

starting_point = (5690, -100, 202)
ball_location = (9115, 3210, 570.456055)
end_point = (5115, 3210, 2400)

ned = np.subtract(end_point, starting_point)/100



client.moveToPositionAsync(ned[0], ned[1], -ned[2], 5).join()

client.hoverAsync().join()

client.simSetCameraOrientation("0", airsim.to_quaternion(-pi/6,0,0))



responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene)])

dir = "airsim_drone"
print ("Saving images to %s" % dir)
try:
    os.makedirs(dir)
except OSError:
    if not os.path.isdir(dir):
        raise

airsim.write_file(dir + '/photo.png', response.image_data_uint8)

################################################################
#       E    N    D
################################################################







client.armDisarm(False)
client.enableApiControl(False)
