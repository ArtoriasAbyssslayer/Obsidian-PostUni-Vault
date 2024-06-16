# **Visual Object Tracking using SiamFC**

Presenter: **Iason Karakostas** - iasonekv@csd.auth.gr

**SiamFC**: [link](https://www.robots.ox.ac.uk/~luca/siamese-fc.html) - [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.pdf) - [github](https://github.com/huanglianghua/siamfc-pytorch)

***



### **Download code and data** 

* Clone the SiamFC-pytorch GitHub repository
* Download an image sequence
* Download pretrained SiamFC weights
* Installed additional Python libraries


```python
# clone SiamFC-PyTorch repo
!git clone https://github.com/huanglianghua/siamfc-pytorch.git
```

    Cloning into 'siamfc-pytorch'...
    remote: Enumerating objects: 104, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 104 (delta 3), reused 0 (delta 0), pack-reused 96[K
    Receiving objects: 100% (104/104), 41.65 KiB | 13.88 MiB/s, done.
    Resolving deltas: 100% (36/36), done.
    


```python
# download a video to perform the tracking task
!wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qiGPGzx2qwECaMPccklv61TUm7t_qEfX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qiGPGzx2qwECaMPccklv61TUm7t_qEfX" -O Crossing.zip && rm -rf /tmp/cookies.txt
!unzip Crossing.zip 
```

    Crossing.zip          0%[                    ]       0  --.-KB/s               Crossing.zip        100%[===================>]   1.38M  --.-KB/s    in 0.008s  
    Archive:  Crossing.zip
       creating: Crossing/
      inflating: Crossing/groundtruth_rect.txt  
      inflating: Crossing/cfg.json       
       creating: Crossing/img/
      inflating: Crossing/images.txt     
      inflating: Crossing/groundtruth.txt  
      inflating: Crossing/attrs.txt      
      inflating: Crossing/img/0071.jpg   
      inflating: Crossing/img/0065.jpg   
      inflating: Crossing/img/0059.jpg   
      inflating: Crossing/img/0105.jpg   
      inflating: Crossing/img/0111.jpg   
      inflating: Crossing/img/0110.jpg   
      inflating: Crossing/img/0104.jpg   
      inflating: Crossing/img/0058.jpg   
      inflating: Crossing/img/0064.jpg   
      inflating: Crossing/img/0070.jpg   
      inflating: Crossing/img/0066.jpg   
      inflating: Crossing/img/0072.jpg   
      inflating: Crossing/img/0099.jpg   
      inflating: Crossing/img/0112.jpg   
      inflating: Crossing/img/0106.jpg   
      inflating: Crossing/img/0107.jpg   
      inflating: Crossing/img/0113.jpg   
      inflating: Crossing/img/0098.jpg   
      inflating: Crossing/img/0073.jpg   
      inflating: Crossing/img/0067.jpg   
      inflating: Crossing/img/0063.jpg   
      inflating: Crossing/img/0077.jpg   
      inflating: Crossing/img/0088.jpg   
      inflating: Crossing/img/0117.jpg   
      inflating: Crossing/img/0103.jpg   
      inflating: Crossing/img/0102.jpg   
      inflating: Crossing/img/0116.jpg   
      inflating: Crossing/img/0089.jpg   
      inflating: Crossing/img/0076.jpg   
      inflating: Crossing/img/0062.jpg   
      inflating: Crossing/img/0048.jpg   
      inflating: Crossing/img/0074.jpg   
      inflating: Crossing/img/0060.jpg   
      inflating: Crossing/img/0100.jpg   
      inflating: Crossing/img/0114.jpg   
      inflating: Crossing/img/0115.jpg   
      inflating: Crossing/img/0101.jpg   
      inflating: Crossing/img/0061.jpg   
      inflating: Crossing/img/0075.jpg   
      inflating: Crossing/img/0049.jpg   
      inflating: Crossing/img/0012.jpg   
      inflating: Crossing/img/0006.jpg   
      inflating: Crossing/img/0007.jpg   
      inflating: Crossing/img/0013.jpg   
      inflating: Crossing/img/0005.jpg   
      inflating: Crossing/img/0011.jpg   
      inflating: Crossing/img/0039.jpg   
      inflating: Crossing/img/0038.jpg   
      inflating: Crossing/img/0010.jpg   
      inflating: Crossing/img/0004.jpg   
      inflating: Crossing/img/0028.jpg   
      inflating: Crossing/img/0014.jpg   
      inflating: Crossing/img/0015.jpg   
      inflating: Crossing/img/0001.jpg   
      inflating: Crossing/img/0029.jpg   
      inflating: Crossing/img/0017.jpg   
      inflating: Crossing/img/0003.jpg   
      inflating: Crossing/img/0002.jpg   
      inflating: Crossing/img/0016.jpg   
      inflating: Crossing/img/0033.jpg   
      inflating: Crossing/img/0027.jpg   
      inflating: Crossing/img/0026.jpg   
      inflating: Crossing/img/0032.jpg   
      inflating: Crossing/img/0024.jpg   
      inflating: Crossing/img/0030.jpg   
      inflating: Crossing/img/0018.jpg   
      inflating: Crossing/img/0019.jpg   
      inflating: Crossing/img/0031.jpg   
      inflating: Crossing/img/0025.jpg   
      inflating: Crossing/img/0009.jpg   
      inflating: Crossing/img/0021.jpg   
      inflating: Crossing/img/0035.jpg   
      inflating: Crossing/img/0034.jpg   
      inflating: Crossing/img/0020.jpg   
      inflating: Crossing/img/0008.jpg   
      inflating: Crossing/img/0036.jpg   
      inflating: Crossing/img/0022.jpg   
      inflating: Crossing/img/0023.jpg   
      inflating: Crossing/img/0037.jpg   
      inflating: Crossing/img/0050.jpg   
      inflating: Crossing/img/0044.jpg   
      inflating: Crossing/img/0078.jpg   
      inflating: Crossing/img/0093.jpg   
      inflating: Crossing/img/0087.jpg   
      inflating: Crossing/img/0118.jpg   
      inflating: Crossing/img/0119.jpg   
      inflating: Crossing/img/0086.jpg   
      inflating: Crossing/img/0092.jpg   
      inflating: Crossing/img/0079.jpg   
      inflating: Crossing/img/0045.jpg   
      inflating: Crossing/img/0051.jpg   
      inflating: Crossing/img/0047.jpg   
      inflating: Crossing/img/0053.jpg   
      inflating: Crossing/img/0084.jpg   
      inflating: Crossing/img/0090.jpg   
      inflating: Crossing/img/0091.jpg   
      inflating: Crossing/img/0085.jpg   
      inflating: Crossing/img/0052.jpg   
      inflating: Crossing/img/0046.jpg   
      inflating: Crossing/img/0042.jpg   
      inflating: Crossing/img/0056.jpg   
      inflating: Crossing/img/0081.jpg   
      inflating: Crossing/img/0095.jpg   
      inflating: Crossing/img/0094.jpg   
      inflating: Crossing/img/0080.jpg   
      inflating: Crossing/img/0057.jpg   
      inflating: Crossing/img/0043.jpg   
      inflating: Crossing/img/0069.jpg   
      inflating: Crossing/img/0055.jpg   
      inflating: Crossing/img/0041.jpg   
      inflating: Crossing/img/0096.jpg   
      inflating: Crossing/img/0082.jpg   
      inflating: Crossing/img/0109.jpg   
      inflating: Crossing/img/0120.jpg   
      inflating: Crossing/img/0108.jpg   
      inflating: Crossing/img/0083.jpg   
      inflating: Crossing/img/0097.jpg   
      inflating: Crossing/img/0040.jpg   
      inflating: Crossing/img/0054.jpg   
      inflating: Crossing/img/0068.jpg   
    


```python
# download pretrained weights
!wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4" -O siamfc_alexnet_e50.pth && rm -rf /tmp/cookies.txt
```

    siamfc_alexnet_e50.   0%[                    ]       0  --.-KB/s               siamfc_alexnet_e50. 100%[===================>]   8.93M  --.-KB/s    in 0.06s   
    


```python
!pip install got10k
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting got10k
      Downloading got10k-0.1.3.tar.gz (31 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from got10k) (1.21.6)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from got10k) (3.2.2)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from got10k) (7.1.2)
    Requirement already satisfied: Shapely in /usr/local/lib/python3.7/dist-packages (from got10k) (1.8.4)
    Collecting fire
      Downloading fire-0.4.0.tar.gz (87 kB)
    [K     |████████████████████████████████| 87 kB 4.6 MB/s 
    [?25hCollecting wget
      Downloading wget-3.2.zip (10 kB)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from fire->got10k) (1.15.0)
    Requirement already satisfied: termcolor in /usr/local/lib/python3.7/dist-packages (from fire->got10k) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->got10k) (0.11.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->got10k) (3.0.9)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->got10k) (1.4.4)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->got10k) (2.8.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->got10k) (4.1.1)
    Building wheels for collected packages: got10k, fire, wget
      Building wheel for got10k (setup.py) ... [?25l[?25hdone
      Created wheel for got10k: filename=got10k-0.1.3-py3-none-any.whl size=43871 sha256=044d4ca0bf700323856075dc788bb45895fa4f1cdac08b9a31c6f17bafb59704
      Stored in directory: /root/.cache/pip/wheels/62/fe/c9/2a3cfd209474f1da9b4a1b2261488fbf55e975be561295874d
      Building wheel for fire (setup.py) ... [?25l[?25hdone
      Created wheel for fire: filename=fire-0.4.0-py2.py3-none-any.whl size=115942 sha256=cc3b52358c88b0e4331785b815e7056e575de3b816c2f4f86b963ea321c03409
      Stored in directory: /root/.cache/pip/wheels/8a/67/fb/2e8a12fa16661b9d5af1f654bd199366799740a85c64981226
      Building wheel for wget (setup.py) ... [?25l[?25hdone
      Created wheel for wget: filename=wget-3.2-py3-none-any.whl size=9675 sha256=dba41c46f4ac6b76040e6b157d4bc5d8d2392c4b1e3c39f1f0236ec05342daa1
      Stored in directory: /root/.cache/pip/wheels/a1/b6/7c/0e63e34eb06634181c63adacca38b79ff8f35c37e3c13e3c02
    Successfully built got10k fire wget
    Installing collected packages: wget, fire, got10k
    Successfully installed fire-0.4.0 got10k-0.1.3 wget-3.2
    

### **Run the code**

As a first step we will import TrackerSiamFC class.


```python
%cd /content/siamfc-pytorch
from siamfc.siamfc import TrackerSiamFC
```

    /content/siamfc-pytorch
    

Now we will execute SiamFC. Since we are not able to play video (at least not easily) on Colab, we will save the tracking results and use OpenCV later in order to create the resulting video.


```python
from __future__ import absolute_import

import os
import glob
import numpy as np


seq_dir = os.path.expanduser('/content/Crossing/')
img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    
net_path = '/content/siamfc_alexnet_e50.pth'
tracker = TrackerSiamFC(net_path=net_path)
boxes, times = tracker.track(img_files, anno[0], visualize=False)

```

### **Visualize the results**


```python
import cv2
import os

# output file
output_file = os.path.join('out.mp4')

# parameters for saving video output
img = cv2.imread(img_files[0])
frame_width = img.shape[1]
frame_height = img.shape[0]
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video = cv2.VideoWriter(output_file, fourcc, float(25), (frame_width, frame_height))

for ii in enumerate(img_files):
    img_name = ii[1]
    box = boxes[ii[0]]
    img = cv2.imread(img_name)
    x, y, w, h = box.astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    video.write(img)

video.release()
  
```
