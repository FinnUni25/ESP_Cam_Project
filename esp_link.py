# hier Code
import cv2
import urllib.request as open_url
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

url = "http://192.168.178.81/capture"
image_dir = "/home/finn/Arduino/ESP32_Cam/images"
time_interval = 0.5
image_count = 20

counter = 0

while counter < 20:
    counter = counter + 1
    print(counter)
    img_raw = open_url.urlopen(url)
    img_np = np.asarray(bytearray(img_raw.read()), dtype="uint8")
    img = cv2.imdecode(img_np, -1)
    # blau und rot sind vertauscht, das ändern:
    #img[:,:,0] = img[:,:,0]*1 #red channel
    #img[:,:,1] = img[:,:,1]*1 #green channel
    #img[:,:,2] = img[:,:,2]*1 #blue channel
    img_rgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

    # plotting for testing
    #matplotlib.use("TkAgg")
    #plt.imshow(img_rgb)
    #plt.axis("off")
    #plt.show()

    file_name = f"{image_dir}/{counter}.jpg"
    cv2.imwrite(file_name, img)
    

    time.sleep(0.5)

