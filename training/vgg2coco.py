import json
import numpy as np
import os
import glob
import cv2

json_file_path = "C:/Users/ITRI/Desktop/Harish/MRCNN_RBP/dataset_TF2_verysmall/val/via_region_data.json"

image_path = os.path.split(json_file_path)
os.chdir(image_path[0])
for file in glob.glob("*.png"): # DEFINE png or Jpeg
    im = cv2.imread(file)
    break

width = im.shape[1]
height = im.shape[0]
final_info ={}
final_info['year'] = 2020
final_info['version'] = "1.0"
final_info['description'] = "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)"
final_info['contributor'] = "" 
final_info['url'] = "http://www.robots.ox.ac.uk/~vgg/software/via/"
final_info['date_created'] = "Fri Jul 17 2020 22:58:40 GMT+0800 (Taipei Standard Time)"
final_license={}
final_license['id'] = 0
final_license['name'] = "Unknown License"
final_license['url'] = ""

with open(json_file_path) as json_file:
    data = json.load(json_file)
    id = 0
    mask_id = 0

    names = []
    final_images=[]
    final_annotation = []
    final_categories = []
    for img in data:
        info = data[img] 
# images =====================================================================
        images = {}           
        images['id'] = id
        
        images['width'] = width
        images['height'] = height
        images['file_name'] = info['filename']
        images['license'] = 0
        images['date_captured'] = ""
        final_images.append(images)
        
# annotations ================================================================
        
        for regi in info['regions']:
            reg = info['regions'][regi]
            anno = {}
            X = reg['shape_attributes']['all_points_x']
            Y = reg['shape_attributes']['all_points_y']
            segment = []
            points = []
            for t in range(len(X)):
                segment.append(X[t])
                segment.append(Y[t])
                points.append([X[t],Y[t]])
            if len(segment)<5:
                continue
            anno['segmentation'] = [segment]
            #calcualte area
            contour = np.array(points)
            x = contour[:, 0]
            y = contour[:, 1]
            if  ((max(X)-min(X)) == 0) or ((max(Y)-min(Y)) == 0):
                continue
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            anno['area'] = area
            anno['bbox'] = [min(X), min(Y),max(X)-min(X),max(Y)-min(Y)]
            anno['iscrowd'] = 0
            anno['id'] = mask_id
            anno['image_id'] = id
           
            
            mask_id += 1
             
            
# categories ================================================================
            categ = {}
            sup_categ = list(reg['region_attributes'])
            name = list(reg['region_attributes'].values())
            if name[0] not in names:
                names.append(name[0])
                categ['supercategory'] = sup_categ[0]
                categ['id'] = names.index(name[0])
                categ['name'] = name[0]
                final_categories.append(categ)
                
            anno['category_id'] = names.index(name[0])
            final_annotation.append(anno)
        id += 1
            
data_coco = {}
data_coco["images"] = final_images
data_coco["categories"] = final_categories
data_coco["annotations"] = final_annotation          
#final = {'info':final_info,'images':final_images, 'annotations':final_annotation, 'licenses':[final_license], 'categories':final_categories}


head = os.path.split(json_file_path)
output_path = head[0]+"/COCO.json"


os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
json.dump(data_coco, open(output_path, "w"), indent=4)
# with open(output_path, 'w') as outfile:
#     json.dump(final, outfile)