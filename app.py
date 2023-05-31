from flask import Flask, request, render_template, jsonify, send_from_directory
import base64
import os
import cv2
import json
from datetime import datetime
import mrcnn.model as modellib
from mrcnn.config import Config
import numpy as np
from skimage.morphology import skeletonize
# from numba import cuda
from keras import backend as K
import tensorflow as tf
import tempfile
import logging
import ToothRoothCalculation as trc
from logging.config import dictConfig
import sys

logging.basicConfig(level=logging.DEBUG)
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# device = cuda.get_current_device()
# device.reset()
K.clear_session()


class OPT_Config(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "kiefer"

    NUM_CLASSES = 5  # Background + zahn + implantat + 8er zahn + sonstiges
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class NumpyArrayEncoder(json.JSONEncoder):
    """ Custom JSON Encoder to Serialize NumPy nd.array """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_obj_data(result):
    class_names = ['BG', 'Zahn', 'Implantat', '8er Zahn', 'Sonstiges']
    roi = result['rois'].tolist()
    class_ids = result['class_ids'].tolist()
    rbb_points, sklet_points = axis_lines(result)

    # remove other class teeth
    teeth_countours = result['contours'].copy()
    teeth_roi = result['rois'].copy().tolist()
    teeth_rbox = result['rbox'].copy()
    for i in range(len(class_ids)):
        if (class_ids[i]!= 1):
            teeth_countours.pop(i)
            teeth_roi.pop(i)
            teeth_rbox.pop(i)

    # sort tooth in sequence
    sorted_tooth_boundary = trc.sort_tooth_lrtb(teeth_countours)
    sorted_roi = trc.sort_tooth_lrtb([[[roi[1], roi[0]], [roi[3], roi[2]]] for roi in teeth_roi])
    sorted_rotated_box = trc.sort_tooth_lrtb(teeth_rbox)

    ## PCA calculation and visualisation
    sorted_pca_min_max = trc.pca_calculation(sorted_tooth_boundary)

    sorted_relative_points = trc.calculate_relative_point(teeth_countours, teeth_roi, teeth_rbox)

    data = []
    teeth_data = []
    for i in range(len(class_ids)):
        data.append({
            "classId": class_ids[i],  # class-ids for 4 classes
            "classIdAsString": class_names[class_ids[i]],  # class names ('Zahn', 'Implantat', '8er Zahn', 'Sonstiges')
            "toothBoundingBox": roi[i],  # normal boundingboxes for each tooth
            "toothShape": result['contours'][i],  # points of tooth contour for every tooth
            "rotatedBoundingBox": result['rbox'][i],  # rotated boundingbox oriented in the direction of ooth axes
            "skelet": result['skelet'][i],  # points of calculated skeleton for every tooth
            "rbboxLine": rbb_points[i],  # calculated tooth axes using rotated boundingboxes
            "skeletLine": sklet_points[i],  # calculated tooth axes using skeletons
        })

    for i in range(len(sorted_roi)):
        teeth_data.append({
            "sortedBoundingBoxe": sorted_roi[i],  # list of bbox points lb and rt of sorted tooth list
            "sortedRotatedBox": sorted_rotated_box[i],  # list of rotated box of sorted tooth list
            "sortedToothShape": sorted_tooth_boundary[i],  # list of points of sorted tooth list
            "sortedRelativeToLB": sorted_relative_points[i],  # relative point to lb of rotated bbox of sorted tooth list
            "sortedMinMaxToothAxis": sorted_pca_min_max[i]  # [min, max] of the tooth axis of sorted tooth list
        })
    return data, teeth_data

# def make_polygon(boxes, masks):
#     """Convert masks to polygons"""
#     N = boxes.shape[0]
#     pboxes = []
#     for i in range(N):
#         mask = masks[:, :, i]
#         padded_mask = np.zeros(
#             (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#         padded_mask[1:-1, 1:-1] = mask
#         contours = find_contours(padded_mask, 0.5)
#         for verts in contours:
#             # Subtract the padding and flip (y, x) to (x, y)
#             verts = np.fliplr(verts) - 1
#             pboxes.append(verts)
#     return pboxes


def return_rotated_box(masks):
    rboxes = []
    cont = []
    skelets = []
    N = masks.shape[-1]
    print('Number of instances: ', N)

    for mask in [masks[:, :, i] for i in range(N)]:
        # convert mask to cv2 image format
        im = np.array(mask * 255, dtype=np.uint8)
        # find the threshold mask
        threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        # find contours of shapes in masks
        # contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL: retrieves only the extreme outer contours.
        contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # find and draw bounding boxes for contours
        # logging.info(f"number of contours: {len(contours)}")

        # cnt = contours[1::2]    # odd contours if retrival mode = cv2.RETR_TREE

        c = sorted(contours, key=len)[-1]
        # for c in contours:
        # if len(c) < 50:  # to avoid making mini bboxes
        #     continue

        hull = cv2.convexHull(c)  # convert to  a convex shape
        rect = cv2.minAreaRect(hull)  # build rotated boxes
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        skeleton = skeletonize(mask, method='zhang')

        # tmp = [p.tolist()for x in c for p in x]
        tmp = [p[0].tolist() for p in c]

        cont.append(tmp)
        rboxes.append(box)
        skelets.append(np.argwhere(skeleton == True))
        # print(skelets)
    return rboxes, cont, skelets


def axis_lines(result):
    import math
    rbb_points = []
    sklet_points = []
    for i in range(len(result['skelet'])):
        # calculating points using skelets
        cordinates = result['skelet'][i].T
        line = np.polyfit(cordinates[1], cordinates[0], 1)
        x = cordinates[1]
        # y = line[0]*x + line[1]

        sklet_points.append(([int(sorted(x)[0]), int(line[0] * sorted(x)[0] + line[1])],
                             [int(sorted(x)[-1]), int(line[0] * sorted(x)[-1] + line[1])]))

        # calculating points using rbboxes
        x_1 = int(result['rbox'][i][0][0] + result['rbox'][i][1][0]) * 0.5
        y_1 = int(result['rbox'][i][0][1] + result['rbox'][i][1][1]) * 0.5
        x_2 = int(result['rbox'][i][2][0] + result['rbox'][i][3][0]) * 0.5
        y_2 = int(result['rbox'][i][2][1] + result['rbox'][i][3][1]) * 0.5
        dist_1 = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        x_3 = int(result['rbox'][i][0][0] + result['rbox'][i][3][0]) * 0.5
        y_3 = int(result['rbox'][i][0][1] + result['rbox'][i][3][1]) * 0.5
        x_4 = int(result['rbox'][i][1][0] + result['rbox'][i][2][0]) * 0.5
        y_4 = int(result['rbox'][i][1][1] + result['rbox'][i][2][1]) * 0.5
        dist_2 = math.sqrt((x_4 - x_3) ** 2 + (y_4 - y_3) ** 2)

        rbb_points.append([[int(x_3), int(y_3)], [int(x_4), int(y_4)]] if dist_2 > dist_1 else [[int(x_1), int(y_1)],
                                                                                                [int(x_2), int(y_2)]])

    return rbb_points, sklet_points


def opg_analysis(image):
    """Load a specific image from the images folder"""
    try:
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        print("Image loaded successfully")
    except:
        print('Skipping unreadable image test.png')
    K.clear_session()
    print("model loading ...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"tmpdir: {tmpdirname}")
        model = modellib.MaskRCNN(mode="inference", model_dir=tmpdirname, config=OPT_Config())
        logging.info("model loaded ...")
        model.load_weights("model/model.h5", by_name=True)
        logging.info("weights loaded ...")
        # Run detection
        results = model.detect([image], verbose=0)
    r = results[0]
    logging.info("Image analysis completed!")
    # contour = make_polygon(r['rois'], r['masks'])
    rotated_box, contour, skelets = return_rotated_box(r['masks'])
    r['contours'] = contour
    r['rbox'] = rotated_box
    r['skelet'] = skelets
    K.clear_session()
    # device.reset()
    return r


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'images/favicon.ico')


@app.route("/ping", methods=['GET'])
def ping():
    now = datetime.now().strftime("%H:%M:%S")
    logging.info(f"time of ping arrival {now}")
    # return (f"Hello! time: {now}")
    return render_template('index.html', data=f"Hello! ping time: {now}")


@app.route("/detect", methods=['POST'])
def detect():
    logging.info("request received")
    logging.info(f"dict keys are: {request.form.to_dict().keys()}")
    if not request.form or 'image' not in request.form:
        logging.info(400)
    else:
        logging.info(f"received image form: {request.form['image'][:50]} ...")
    # get the base64 encoded string
    for k in request.form.to_dict().keys():
        logging.info(f"received image form from key {k}: {request.form[k][:50]} ...")
    try:
        im_b64 = request.form["image"]
        logging.info(f"Received image is:  {im_b64[:50]} ...")
    except ValueError:
        logging.info("Oops! That was no valid image data.  Try again...")
    logging.info("request delivered")
    # converting base64 string to numpy array image
    img_data = base64.b64decode(im_b64)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logging.info("Analysis started ...")
    # calculating results
    result = opg_analysis(img_np)
    data, teeth_data = get_obj_data(result)
    logging.info("Analysis Done!")
    # result_dict = {'result': json.dumps(data, cls=NumpyArrayEncoder)}
    # device.reset()
    K.clear_session()
    return json.dumps(data, cls=NumpyArrayEncoder)

@app.route("/detectToothRoot", methods=["POST"])
def detectToothRoot():
    print("Ping successful!")
    try:
        request_json = request.get_json()
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content": exception_message})


    print("Recieved data: ", request_json.keys())
    image = request_json['image']
    logging.info("request delivered")
    # converting base64 string to numpy array image
    img_data = base64.b64decode(image)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logging.info("Analysis started ...")
    # calculating results
    result = opg_analysis(img_np)
    data, teeth_data = get_obj_data(result)
    logging.info("Analysis Done!")
    # result_dict = {'result': json.dumps(data, cls=NumpyArrayEncoder)}
    # device.reset()
    K.clear_session()
    # return json.dumps({'data': data,
    #                   'teeth_data': teeth_data}, cls=NumpyArrayEncoder)
    return json.dumps(teeth_data, cls=NumpyArrayEncoder)

if __name__ == '__main__':
    # from waitress import serve

    app.run(debug=False, host='0.0.0.0', port='8000')
    # app.run(debug=True)
    logging.info(" Server is running ...")
    # serve(app, host='0.0.0.0', port='8000')
