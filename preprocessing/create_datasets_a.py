import cv2
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import os
import shutil
from pathlib import Path

repo_path = Path(__file__).resolve().parent.parent

random.seed(357)
np.random.seed(357)

BG_IMAGE_DIR = repo_path / "bfmc_data" / "base" / "datasets_bg"/ "datasets_a" / "images"
BG_LABEL_DIR = repo_path / "bfmc_data" / "base" / "datasets_bg"/ "datasets_a" / "labels"
DEST_IMAGE_DIR = repo_path / "bfmc_data" / "base" / "datasets" / "datasets_a" / "images"
DEST_LABEL_DIR = repo_path / "bfmc_data" / "base" / "datasets" / "datasets_a" / "labels"

os.makedirs(DEST_IMAGE_DIR, exist_ok=True)
os.makedirs(DEST_LABEL_DIR, exist_ok=True)
for filename in os.listdir(BG_LABEL_DIR):
    if filename.endswith(".txt"):
        src = os.path.join(BG_LABEL_DIR, filename)
        dst = os.path.join(DEST_LABEL_DIR, filename)
        shutil.copyfile(src, dst)

def compute_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    return interWidth * interHeight

def overlap(boxA, boxB, threshold=0.3):
    inter_area = compute_intersection_area(boxA, boxB)
    areaA = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2])
    areaB = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2])

    if areaA == 0 or areaB == 0:
        return False

    return (inter_area / areaA > threshold) or (inter_area / areaB > threshold)
def insertImage(imgs, imgClasses, ratios, bg, imgType, number, df, testing = False):
    name = df.iloc[number,0].replace("'", "")
    name2 = name.replace("car_ims/", "")
    # print("number, imgclasses, name: ", number, imgClasses, name2)
    name_txt = name.replace("jpg", "txt")
    name_txt = name_txt.replace("car_ims/", "")
    label_path = os.path.join(DEST_LABEL_DIR, name_txt)
    f = open(label_path, 'a')
    df1 = pd.read_csv(label_path, sep=' ', header=None)
    _, x_c, y_c, w_n, h_n = df1.iloc[0, 0:5]
    img_h, img_w = bg.shape[:2]
    x_center = x_c * img_w
    y_center = y_c * img_h
    w_pix    = w_n * img_w
    h_pix    = h_n * img_h
    left1   = int(x_center - w_pix/2)
    right1  = int(x_center + w_pix/2)
    top1    = int(y_center - h_pix/2)
    bot1    = int(y_center + h_pix/2)
    ds = np.array([int(left1), int(right1), int(top1), int(bot1)])
    # print("ds: ", ds)
    f = open(label_path, 'a')
    occupied = [ds]
    for i in range(len(imgs)):
        stuck = False
        t1 = time.time()+10
        if testing:
            print("i: ",i)
        img1 = imgs[i]
        flag = True
        while flag:
            if time.time()>t1:
                stuck = True
                print("stuck")
                break
            flag = False
            sz = max(img1.shape)
            img = img1
            #print("shapes: ", bg.shape, img.shape)
            maxOffsetX = bg.shape[1] - img.shape[1]
            maxOffsetY = bg.shape[0] - img.shape[0]

            offsetX = np.random.randint(0, high=maxOffsetX)
            offsetY = np.random.randint(0, high=maxOffsetY)
            left,right,top,bot=offsetX,offsetX+img.shape[1],offsetY,offsetY+img.shape[0]
            dimensions = np.array([left, right, top, bot])
            #print("dimensions: ", dimensions)
            for o in occupied:
                if overlap(dimensions, o):
                    # print("overlapped, dimensions: ", dimensions, "o: ", o)
                    flag = True
        if stuck:
            print("stuck at i: ", i)
            continue
        d = dimensions.copy()
        occupied.append(d)
        bg[top:bot,left:right] = img
        txt_param1 = str((left+right)/2/bg.shape[1])
        txt_param2 = str((top+bot)/2/bg.shape[0])
        txt_param3= str(img.shape[1]/bg.shape[1])
        txt_param4= str(img.shape[0]/bg.shape[0])        
        txt_line = (' '+str(imgClasses[i])+' '+str((left+right)/2/bg.shape[1])+' '+str((top+bot)/2/bg.shape[0])
                +' '+str(img.shape[1]/bg.shape[1])+' '+str(img.shape[0]/bg.shape[0])+'\n')
        if testing:
            print('bg, img: ', bg.shape,img.shape)
            print('maxOffsetX,maxOffsetY,boundaryX: ',maxOffsetX,maxOffsetY,bg.shape[1]/3*(i+1))
            print('offsetX,offsetY: ',offsetX,offsetY)
            print('top,bot,left,right :',top,bot,left,right)
            print('left,right,img.shape[1]: ',left,right,bg.shape[1])
            cv2.rectangle(bg, (left, top), (right, bot), (0, 0, 255), thickness=2)
            print('txt_line: '+txt_line)
            print("saving...")
        #cv2.rectangle(bg, (left, top), (right, bot), (0, 0, 255), thickness=2)
        f.write(txt_line)
    image_save_path = os.path.join(DEST_IMAGE_DIR, name2)
    cv2.imwrite(image_save_path, bg)
    if testing:
        cv2.imshow('1', bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    f.close()
    return bg
def create(imgClasses, imgIndices, number ,imgType, flag, df):
    roadNum = number
    name = df.iloc[number,0].replace("'", "")
    name = name.replace("car_ims/","")
    bg_path = os.path.join(BG_IMAGE_DIR, name)
    bg = cv2.imread(bg_path)
    names = []
    imgs = []
    for i in range(num):
        img_path = repo_path / "bfmc_data" / "generated" / "crop_augmented_resized" / str(int(imgClasses[i])) / f"{int(imgIndices[i])}.jpg"
        names.append(img_path)
        img = cv2.imread(img_path)
        imgs.append(img)
    isNone = False
    for i in range(num):
        if imgs[i] is None:
            print("None: ", names[i])
            isNone = True
            exit()
    if isNone:
        exit()
    ratios = np.random.uniform(0.10,0.73, 4)
    bg=insertImage(imgs,imgClasses, ratios, bg, imgType, number, df, testing=flag)
    return bg

def process_item(i, classCounts, classIndices, df):
    # Uniformly choose 'num' images from the entire dataset.
    global total  # total is defined later as the sum of classCounts
    global num    # number of images to choose, already defined
    
    global_indices = np.random.randint(0, total, size=num)
    imgClasses = []
    imgIndices = []
    
    # Map global index to a specific class and its local index.
    for gi in global_indices:
        cls = np.searchsorted(classIndices, gi, side='right')
        local_index = gi - (classIndices[cls-1] if cls > 0 else 0)
        imgClasses.append(cls)
        # Adjust by adding 1 if your image naming starts at 1
        imgIndices.append(local_index + 1)
    
    imgClasses = np.array(imgClasses)
    imgIndices = np.array(imgIndices)
    
    for j in imgClasses:
        IdxCount[int(j)] += 1
    create(imgClasses, imgIndices, i, 'train', False, df)

t2 = time.time()
num = 3
cropped_base_dir = repo_path / "bfmc_data" / "generated" / "crop_augmented_resized"
classFolders = sorted(
    [f for f in os.listdir(cropped_base_dir) if f.isdigit()], 
    key=lambda x: int(x)
)

numClasses = len(classFolders)
classCounts = np.array([
    len([f for f in os.listdir(os.path.join(cropped_base_dir, cls)) if f.lower().endswith(('.jpg', '.png'))])
    for cls in classFolders
])

classIndices = np.cumsum(classCounts)
total = np.sum(classCounts)

unique_numbers = random.sample(range(total), total)
np.save((repo_path / "bfmc_data" / "generated" / "unique_numbers.npy"), unique_numbers)
print(f"Saved unique_numbers.npy with {len(unique_numbers)} entries")

print("classFolders:", classFolders)
print("classIndices: ", classIndices)
print("classCounts: ", classCounts)
IdxCount = np.zeros(numClasses)
filename2 = repo_path / "bfmc_data" / "base" / 'datasets_bg' / 'datasets_a' / 'annotations.txt'
df = pd.read_csv(filename2, sep='\t', header = None)

with ThreadPoolExecutor() as executor:
    executor.map(process_item, range(16185), [classCounts]*16185, [classIndices]*16185, [df]*16185)

print("Completed processing items")
print("IdxCount: ", IdxCount)
print("time: ", time.time()-t2)

    