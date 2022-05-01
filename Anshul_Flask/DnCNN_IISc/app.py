import os
import tensorflow as tf
import subprocess
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for,request, send_from_directory, redirect
#from wce_main import classification
#from covid import get_mask_rcnn_model, inference
import shutil
import cv2
import numpy as np
import re
from infer_sarnet import Denoiser_DnCNN
from infer_sarnet import Thresholding
from Gcds import GCDS
import skimage


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # or '0,1', or '0,1,2' or '1,2,3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')

print('Visible Physical Devices: ',physical_devices)

for gpu in physical_devices:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.threading.set_inter_op_parallelism_threads(32)

tf.config.threading.set_intra_op_parallelism_threads(32)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.bool_)
    y_pred = np.asarray(y_pred).astype(np.bool_)
    
    tp = np.logical_and(y_true , y_pred).sum()
    tn = np.logical_and((1 - y_true), (1 - y_pred)).sum()
    fp = np.logical_and((1 - y_true), y_pred).sum()
    fn = np.logical_and(y_true, (1 - y_pred)).sum()
    
    sens = tp/(tp + fn)
    spec = tn/(tn + fp)
    dc2 = 2*tp/((2*tp) + fp + fn)
    ji = dc2/(2 - dc2)
    return round(dc2, 3), round(ji, 3), round(sens, 3), round(spec, 3)

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]
    
    
def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.5):
    img = img.astype(np.uint8)
    canvas = img.copy() 
    inst_map = np.array(mask==255, np.uint8)
    if np.mean(inst_map)==0:
        return canvas
    else:
        y1, y2, x1, x2  = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1 
        x1 = x1 - 2 if x1 - 2 >= 0 else x1 
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2 
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2 
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours,_ = cv2.findContours(inst_map_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        cv2.drawContours(inst_canvas_crop, contours, -1, color, -1)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
        out = cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, canvas)
        return out    

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/WCE.html', methods = ['GET','POST'])
def WCE():
    if request.method == 'POST':
        patient_id = request.form["p_id"]
        model_name = request.form["models"]

        print(patient_id, 'patient_id')
        prepend = os.getcwd()
        local_file_path= f'{prepend}/patients/{patient_id}/'
        if os.path.exists(os.path.join(local_file_path))!=True:
            os.mkdir(os.path.join(local_file_path))
        if os.path.exists(os.path.join(local_file_path, "Images"))!=True:
            os.mkdir(os.path.join(local_file_path, "Images"))

        Dict = { }
        op_all = []
        for fn in request.files.getlist('file'):
            filename = secure_filename(fn.filename)       
            input_path= os.path.join(local_file_path, 'Images/' + filename) 
            fn.save(input_path)
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER + 'patients/' + patient_id + '/'
            op_path2 = UPLOAD_FOLDER + 'patients/' + patient_id + '/'

            if os.path.exists(op_path2)!=True:
                os.mkdir(op_path2)

            if model_name == 'DnCNN':
                op_all.append(Denoiser_DnCNN(input_image_path = input_path, output_path = op_path2))
            elif model_name == 'Thresholding':
                op_all.append(Thresholding(input_image_path = input_path, output_path = op_path2))

            elif model_name == 'GCDS':
                op_all.append(GCDS(input_image_path = input_path, output_path = op_path2))

        # endoscopy._normal_abnormal()
        # endoscopy._abnormality()
        
        

        
        
        op_htmls = []

        for op_ in op_all:
            img_id = op_['file']
            psnr = op_['psnr']
            # gt_met = 'static/Labels/' + img_id + 'm.png'
            # gt_ = skimage.io.imread(gt_met)[52:308, 52:308]
            # dc, aji, sens, spec = metrics(gt_, pred)
            # img_ = skimage.io.imread(os.path.join(app.config['UPLOAD_FOLDER'] , img_id + '.png'))
            # gt_overlay = mask_color_img(img_, gt_)
            # skimage.io.imsave(os.path.join(app.config['UPLOAD_FOLDER'], img_id + 'm.png'), gt_overlay)
            # prediction = op_['path'].split('/')[-1]
            # actual = re.split('(\d+)',img_id.split('.')[0])[0]
            op_htmls.append({'inputimage' : os.path.join(app.config['UPLOAD_FOLDER'] , img_id + '.jpg'),
            'predimage' : os.path.join(app.config['UPLOAD_FOLDER'] , img_id + '_pred.jpg'),
            'PSNR': psnr})   

        return render_template('slideshow_new_load.html', links=op_htmls)        
    return render_template("WCE.html")

# @app.route('/COVID19.html', methods=['POST', 'GET'])
# def COVID():
#     if request.method == 'POST':
#         patient_id = request.form["p_id"]
#         file = request.files['file']
        
#         # Input image name 
#         filename = secure_filename(file.filename)
#         # Predicted image name 
#         predimage_name = filename.split('.')[0] + "_mask." + filename.split('.')[1]
#         # Patient directory 
#         path = os.path.join(app.config['UPLOAD_FOLDER'], 'covid', patient_id)
#         # Input image path 
#         inputimage = os.path.join(path, filename) 
#         # Predicted image path 
#         predimage = os.path.join(path, predimage_name)
#         # Make the directories
#         os.makedirs(f"{path}")
#         # Save the uploaded input image 
#         file.save(inputimage)
#         # Predict the mask 
#         inference(inputimage, predimage)
#         # Return the template with the input image and predicted mask 
#         return render_template('CovidPrediction.html', inputimg = inputimage, predimg =  predimage)
#     return render_template("COVID19.html")

# @app.route('/display/<filename>')
# def display_image(filename):
# 	return redirect(url_for('static', filename = filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)