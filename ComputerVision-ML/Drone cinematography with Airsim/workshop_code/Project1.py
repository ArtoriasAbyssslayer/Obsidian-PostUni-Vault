import setup_path
import airsim
import numpy as np


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()




################################################################
#       MOVE INSIDE THE YELLOW BALL
################################################################


world_origin = (5690, -100, 202)
end_point = (9115, 3210, 570.456055)

ned = tuple(np.subtract(end_point, world_origin)/100)


client.moveToPositionAsync(ned[0], ned[1], -ned[2], 5).join()

client.hoverAsync().join()


################################################################
#       E    N    D
################################################################






client.armDisarm(False)
client.enableApiControl(False)
