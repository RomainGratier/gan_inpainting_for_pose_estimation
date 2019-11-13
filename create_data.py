import os
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import PIL.Image
from PIL import Image
import cv2
import skimage.io as io
from utils_dataset import make_grid, from_torch_img_to_numpy, generate, load_occluders, occlude_with_objects, paste_over, resize_by_factor, list_filepaths

dataDir='datasets/coco'
dataType='val2017'
annFile_full='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annFile_human_pose = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)

#occluders = load_occluders('synthetic-occlusion/VOCdevkit/VOC2012')
import pickle

#with open('occluders.pkl', 'wb') as f:
#    pickle.dump(occluders, f)
    
with open('pre_process_folder/occluders.pkl', 'rb') as f:
    occluders = pickle.load(f)
    
occlusion_list = []

for i in [7, 34, 37, 139, 153, 247, 249, 276, 312, 402, 435, 446, 449, 479]:
    occlusion_list.append[occluders[i]]
    
    
def create_folder(name):
    os.makedirs(name, exist_ok=True)
    
def save_img_jpg(arr, img_id, annIds, img_directory):
    im = Image.fromarray(arr)
    im.save(f"{os.path.join(img_directory, str(img_id)+'.'+str(annIds[-1]))}.jpg")

def main(dataDir, dataType, annFile_full, annFile_human_pose, occluders):
    ''' Get the data '''
    # initialize COCO api for instance annotations
    coco=COCO(annFile_full)
    coco_kps=COCO(annFile_human_pose)

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    
    img_directory_occluded = 'occ_img'
    create_folder(img_directory_occluded) 
    
    img_directory_generate = 'gen_img'
    create_folder(img_directory_generate) 
    
    img_directory_real = 'real_img'
    create_folder(img_directory_real) 

    for img_id in tqdm(imgIds):
        # Get img from id
        print()
        img = coco.loadImgs(img_id)[0]

        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        print('idssssssssssssss')
        print(img_id)
        print(annIds)
        print(img['id'])
        pass
        if len(annIds) != 1:
            continue
    
        anns = coco_kps.loadAnns(annIds)
        if anns[0]['area'] < 3000:
            continue
    
        if anns[0]['num_keypoints'] < 10:
            continue
    
        arr = io.imread(img['coco_url'])
        
        gen_arr = arr.copy()
        occ_arr = arr.copy()

        # Slice Bounding Box
        bbox = anns[0]['bbox']
        bbox = list(map(int, bbox))
        bbox_arr = arr[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]

        occluded_im, start_pt, end_pt, center = occlude_with_objects(bbox_arr, occluders)

        # Crop the img where an occlusion appears
        x_y_occ_mask = [end_pt[0]-start_pt[0], end_pt[1]-start_pt[1]]
        center_arr = np.round(center).astype(np.int32)
        c_main_img =[center_arr[0]+bbox[0], center_arr[1]+bbox[1]]
        alpha = 5
        x_y_mask = [c_main_img[1] - x_y_occ_mask[1] - alpha, c_main_img[0] - x_y_occ_mask[0]- alpha]

        occ_arr[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :] = occluded_im

        print(' --------------------------- CROPPING ---------------------------')
        max_crop_side = max(x_y_occ_mask)
        img_for_generation = arr[x_y_mask[0]:x_y_mask[0]+max_crop_side*2 + alpha, x_y_mask[1]:x_y_mask[1]+max_crop_side*2 + alpha, :]
        
        if (img_for_generation.shape[0] == 0) or (img_for_generation.shape[1] == 0):
            print(' --------------------------- CONTINUE --------------------------- ')
            continue
        
        gen_img = generate(img_for_generation, 'generative-inpainting-pytorch/examples/center_mask_256.png', '')
        final_gen_img = cv2.resize(gen_img, (img_for_generation.shape[1],img_for_generation.shape[0]))
        gen_arr[x_y_mask[0]:x_y_mask[0]+max_crop_side*2 + alpha, x_y_mask[1]:x_y_mask[1]+max_crop_side*2 + alpha, :] = final_gen_img
        
        print(' --------------------------- Saving ---------------------------')
        # Save images
        save_img_jpg(occ_arr, img_id, annIds, img_directory_occluded)
        save_img_jpg(gen_arr, img_id, annIds, img_directory_generate)
        save_img_jpg(arr, img_id, annIds, img_directory_real)

                
if __name__ == '__main__':
    main(dataDir, dataType, annFile_full, annFile_human_pose, occlusion_list)
 