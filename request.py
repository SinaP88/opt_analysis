import math

import requests
import base64
import json
import numpy as np
import os
from keras import backend as K
import matplotlib.pyplot as plt
import ToothRoothCalculation as trc

K.clear_session()


def visualize_original_image(IMAGE_DIR):
    import matplotlib.pyplot as plt
    import cv2
    image = cv2.imread(IMAGE_DIR, cv2.IMREAD_COLOR)
    fig2, ax = plt.subplots(1, figsize=(image.shape[1] * 0.01, image.shape[0] * 0.01))
    # fig, ax = plt.subplots(1)
    ax.set_axis_off()  ##########################################
    # auto_show = True
    fig2.tight_layout()

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.imshow(image.astype(np.uint8))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    plt.savefig('results/07_case_OPG.JPG', dpi=235)


def postTestImageJson(url='http://localhost:8000/detect', testImagePath="static/test.png"):
    """ to send a request in json dict format """
    with open(testImagePath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode('utf-8')

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    payload = json.dumps({"image": req_file, "other_key": "value"})
    response = requests.post(url, data=payload, headers=headers)

    print('request sent!', response)
    try:
        data = response.json()

    except requests.exceptions.RequestException:
        print(response.text)
    for i in range(len(json.loads(data['result']))):
        print(f'The bounding box exported from tooth {i} is :', json.loads(data['result'])[str(i)]['toothBoundingBox'])
        print('shape:', json.loads(data['result'])[str(i)]['toothShape'])


def postTestImageMultiDict(url='http://localhost:8000/detect', testImagePath="static/images/test.png"):
    """ to send a request in html url-encoded MultiDict format """
    from werkzeug.datastructures import ImmutableMultiDict
    with open(testImagePath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        req_file = encoded_image.decode('utf-8')

    headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'text/html'}
    payload = ImmutableMultiDict([('image', req_file)])

    response = requests.post(url, data=payload, headers=headers)  # form: for html form, data: for local data

    print('request sent!', response)
    try:
        data = response.json()

    except requests.exceptions.RequestException:
        print(response.text)
    for i in range(len(data)):
        # print(f'The bounding box exported from tooth {i} is :', data[i]['toothBoundingBox'])
        # print('shape:', data[i]['toothShape'])
        print(f'Tooth {i + 1} contour length: ', len(data[i]['toothShape']))

    # Visualize and save the result
    try:
        import cv2
        image = cv2.imread(testImagePath, cv2.IMREAD_COLOR)
        print("Image loaded successfully")
    except:
        print('Skipping unreadable image test.png')
    class_names = ['BG', 'Zahn', 'Implantat', '8er Zahn', 'Sonstiges']
    r = dict()
    r['rois'] = [x['toothBoundingBox'] for x in data]
    r['contour'] = [x['toothShape'] for x in data]
    r['class_id'] = [x['classId'] for x in data]
    r['rbox'] = [x['rotatedBoundingBox'] for x in data]
    r['skeletLine'] = [x['skeletLine'] for x in data]
    r['skelet'] = [x['skelet'] for x in data]
    # sort tooth in sequence
    sorted_tooth_boundary = [x['sortedToothShape'] for x in data]
    sorted_roi = [x['sortedBoundingBoxe'] for x in data]
    sorted_rotated_box = [x['sortedRotatedBox'] for x in data]
    sorted_pca_min_max = [x['sortedMinMaxToothAxis'] for x in data]
    sorted_relative_points = [x['sortedRelativeToLB'] for x in data]


    print("Relative points: ", sorted_relative_points)

    trc.pca_calculation_visualize(sorted_tooth_boundary, sorted_rotated_box)

    # ## visualisation
    # for c in r['contour']:
    #     cv2.polylines(image, [np.array(c).reshape((-1, 1, 2))], True, (0, 255, 0), 1)
    # ## add rotated boxes to image
    # for b in r['rbox']:
    #     cv2.polylines(image, [np.array(b).reshape((-1, 1, 2))], True, (255, 0, 0), 1)
    # ## add bounding boxes to image
    # for a in r['rois']:
    #     cv2.rectangle(image, (a[1], a[0]), (a[3], a[2]), (0, 0, 255), 1)
    # for e in r['skelet']:
    #     # ee = np.array(e).reshape((-1, 1, 2))
    #     eee = np.array(e)
    #     e_invert = np.array([[y, x] for [x, y] in eee])
    #     cv2.polylines(image, [e_invert], False, (255, 100, 10), 2)
    # cv2.imshow('Analysed_OPG.JPG', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('Analysed_OPG.JPG', image)

    ### extract and save rotated boxes to path
    save_path = "OPG"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sorted_rotated_box = trc.sort_tooth_lrtb(r['rbox'])
    for i, b in enumerate(sorted_rotated_box):
        # warped_image = trc.extract_tooth_images(b, image)
        # cv2.imwrite(f"{save_path}/{i + 1}.png", warped_image)

        crop, rotated = trc.crop_rect(image, b)
        cv2.imwrite(f"{save_path}/U{i + 1}.png", crop)
        # cv2.imwrite(f"{save_path}/r{i + 1}.png", rotated)


if __name__ == "__main__":
    testImagePath = "static/images/JB.jpg"
    # visualize_original_image(testImagePath)
    # postTestImageJson(testImagePath=testImagePath)
    postTestImageMultiDict(url='http://192.168.178.65:8000/detect', testImagePath=testImagePath)
    # postTestImageMultiDict(url="http://dev1.orcadent.de:9423/detect", testImagePath=testImagePath)
    # postTestImageMultiDict(url="https://tai-opt10.orcadent.de/detect", testImagePath=testImagePath)
