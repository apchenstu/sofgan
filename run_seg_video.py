import os,cv2,dlib
import numpy as np
import argparse

import torch
from PIL import Image
from modules.BiSeNet import BiSeNet
import torchvision.transforms as transforms
'''
# sample
# -i /path/to/video.mp4 -o xxx.mp4
'''

def initFaceParsing():
    net = BiSeNet(n_classes=20)
    net.cuda()
    net.load_state_dict(torch.load('modules/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),

    ])
    return net,to_tensor

from skimage import morphology
def filtting(label_img):
    res_img = np.zeros(label_img.shape, dtype='uint8')
    labels = np.unique(label_img)

    for item in labels:
        area = label_img == item
        count = area.sum()
        area = morphology.remove_small_holes(area, count // 10)
        area = morphology.remove_small_objects(area, count//10)
        res_img[area > 0] = item
    return res_img

def inpaint(label_img):
     return morphology.area_closing(label_img,area_threshold=2000)

def parsing_img(bisNet, img, to_tensor):
    with torch.no_grad():
        image = Image.fromarray(img)
        image = image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = bisNet(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # parsing = inpaint(filtting(parsing))
    return parsing

remap_list =np.array([0,1,2,2,3,3,4,5,6,7,8,9,9,10,11,12,13,14,15,16])
def id_remap(seg):
    #['background'0,'skin'1, 'l_brow'2, 'r_brow'3, 'l_eye'4, 'r_eye'5,'r_nose'6, 'l_nose'7, 'mouth'8, 'u_lip'9,
    # 'l_lip'10, 'l_ear'11, 'r_ear'12, 'ear_r'13, 'eye_g'14, 'neck'15, 'neck_l'16, 'cloth'17, 'hair'18, 'hat'19]
    return remap_list[seg]

remap_raw_to_new_list = np.array([0,1,2,3,4,5,14,11,12,13,6,7,8,9,10,15,16,17,18,19])
def id_raw_to_new(seg):
    return remap_raw_to_new_list[seg]

def vis_condition_img(img):
    part_colors = [[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],#'skin',1 'eye_brow'2,  'eye'3
                    [240, 157, 240], [255, 212, 255], #'r_nose'4, 'l_nose'5
                    [31, 162, 230], [127, 255, 255], [127, 255, 255],#'mouth'6, 'u_lip'7,'l_lip'8
                    [0, 255, 85], [0, 255, 170], #'ear'9 'ear_r'10
                    [255, 255, 170],
                    [127, 170, 255], [85, 0, 255], [255, 170, 127], #'neck'11, 'neck_l'12, 'cloth'13
                    [212, 127, 255], [0, 170, 255],#, 'hair'14, 'hat'15
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]
    H,W = img.shape
    condition_img_color = np.zeros((H,W,3))

    num_of_class = int(np.max(img))
    for pi in range(1, num_of_class + 1):
        index = np.where(img == pi)
        condition_img_color[index[0], index[1],:] = part_colors[pi]
    return condition_img_color

from scipy.spatial import distance as dist
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def parse_68(landmark) -> list:
    # input shape.part object
    # output: three bboxes, left, right eye and mouth

    left_eye, right_eye, mouth = [], [], []
    for it, seq in enumerate(range(36, 42)):
        left_eye.append(np.array([landmark(seq).x, landmark(seq).y]))

    for it, seq in enumerate(range(42, 48)):
        right_eye.append(np.array([landmark(seq).x, landmark(seq).y]))

    for it, seq in enumerate(range(60, 68)):
        mouth.append(np.array([landmark(seq).x, landmark(seq).y]))

    mouth, left_eye, right_eye = np.stack(mouth), np.stack(left_eye), np.stack(right_eye)
    out = []
    out.append(left_eye)
    out.append(right_eye)
    out.append(mouth)

    return out

def main(args):
    if not os.path.isdir(args.input):
        files = [os.path.basename(args.input)]
        args.input = os.path.dirname(args.input)
    else:
        files = sorted(os.listdir(args.input))

    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()

    # FlowNet2 = initFlowNet2()
    bisNet, to_tensor = initFaceParsing()
    for ind, item in enumerate(files[:-1]):

        path = os.path.join(args.input,item)
        if os.path.isdir(path):
            continue

        cap = cv2.VideoCapture(path)
        if args.save_as_file:
            path_out = os.path.join(args.output, item[:-4] + '_seg')
            if not os.path.exists(path_out):
                os.mkdir(path_out)

        path_out_vis = os.path.join(args.output, 'vis')
        if not os.path.exists(path_out_vis):
            os.mkdir(path_out_vis)
        path_out_vis = os.path.join(path_out_vis, item[:-4] + '.avi')
        out = cv2.VideoWriter(path_out_vis, cv2.VideoWriter_fourcc(*'XVID'), 20,(512, 512))

        count,flow_test_size = 0, 512
        success, img = cap.read()
        prvs_bbox = None
        while success:
            img = cv2.resize(img, (512, 512))


            seg = parsing_img(bisNet, img[..., ::-1], to_tensor)
            seg = np.round(seg).astype('uint8')

            # landmark for eye
            dets = detector(img, 1)
            if len(dets)>0:
                seg[seg==4],seg[seg==5] = 1,1
                shape = predictor(img, dets[0])
                bboxes = parse_68(shape.part)
                if prvs_bbox is not None:
                    for k in range(len(bboxes)):
                        ratio = eye_aspect_ratio(bboxes[k])
                        if ratio < 0.18:
                            continue
                        diff = np.abs(bboxes[k]-prvs_bbox[k])**2
                        diff = diff / max(np.max(diff),1.0)*0.8
                        bboxes[k] = np.round(diff*bboxes[k] + (1.0-diff)*prvs_bbox[k]).astype('int')


                for k,ind in enumerate([4,5]):
                    cv2.fillConvexPoly(seg, bboxes[k], ind)


            # seg = morphology.area_closing(seg, area_threshold=10000)
            seg = seg[20:512-20,20:512-20]
            seg = cv2.resize(seg,(512,512),interpolation=cv2.INTER_NEAREST)
            seg_vis = vis_condition_img(id_remap(seg))[...,::-1]
            out.write(seg_vis.astype('uint8'))
            if args.save_as_file:
                cv2.imwrite(os.path.join(path_out,'%05d.png'%count),seg.astype('uint8'))
            success, img = cap.read()
            prvs_bbox = bboxes
            count += 1
        cap.release()
        out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='render positionMap from mesh')
    parser.add_argument('-i', '--input', type=str, help='obj path or folder')
    parser.add_argument('-o', '--output', type=str, help='folder')
    parser.add_argument('--save_as_file', action='store_true')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    main(args)
