import cv2,os,argparse,pickle,re
import numpy as np

def str_to_time(time):
    # '%H_%M_%S_%f'
    time = re.split(':|_|\.', time)
    return int(time[0])*60*60*1000 + int(time[1])*60*1000 + int(time[2])*1000 + int(time[3])//1000


def convert_stroke(mat_img,stroke_point):
    if len(stroke_point) == 5:
        color = stroke_point['label']
        cv2.line(mat_img, stroke_point['prev'], stroke_point['curr'], color,
                 stroke_point['brush_size'])
    else:
        print("wrong stroke")


def convert_rec(mat_img,rectangle):
    if len(rectangle) == 5:
        color = rectangle['label']
        cv2.rectangle(mat_img, rectangle['prev'], rectangle['curr'],color, -1)
    else:
        print("wrong rectangle")


def convert_fill(mat_img,fill):
    if len(fill) == 4:
        color = fill['label']
        cv2.drawContours(mat_img, fill['contours'], fill['contours_num'], color, -1)
    else:
        print("wrong fill")

def render(img, keys, frames):
    for key in keys:
        if 'Rec' == frames[key]['shape']:
            convert_rec(img, frames[key])
        elif 'Stroke' == frames[key]['shape']:
            convert_stroke(img, frames[key])
        elif 'Fill' == frames[key]['shape']:
            convert_fill(img, frames[key])
    return img

def segmap2rgb( img):
    part_colors = [[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],  # 'skin',1 'eye_brow'2,  'eye'3
                   [240, 157, 240], [255, 212, 255],  # 'r_nose'4, 'l_nose'5
                   [89, 64, 92], [237, 102, 99], [181, 43, 101],  # 'mouth'6, 'u_lip'7,'l_lip'8
                   [0, 255, 85], [0, 255, 170],  # 'ear'9 'ear_r'10
                   [255, 255, 170],
                   [127, 170, 255], [85, 0, 255], [255, 170, 127],  # 'neck'11, 'neck_l'12, 'cloth'13
                   [212, 127, 255], [0, 170, 255],  # , 'hair'14, 'hat'15
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]
    H, W = img.shape
    condition_img_color = np.zeros((H, W, 3)).astype('uint8')

    num_of_class = int(np.max(img))
    for pi in range(1, num_of_class + 1):
        index = np.where(img == pi)
        condition_img_color[index[0], index[1], :] = part_colors[pi]
    return condition_img_color

def main(args):
    if not os.path.exists(args.input):
        print('Could not file input file.')
        exit()

    name = os.path.basename(args.input)
    startTime = str_to_time(name)

    os.makedirs(args.output, exist_ok=True)
    with open(args.input, 'rb') as f:
        frames = pickle.load(f)

    frameIDs = []
    keys = sorted(frames.keys())
    for item in keys:
        frameID = np.round((str_to_time(item) - startTime) / (1000.0 / args.fps)).astype('int')
        frameIDs.append(frameID)

    key_log = []
    img = np.zeros((512,512)).astype('uint8')
    for i, key in enumerate(keys):
        if 1 == len(frames[key].keys()):
            for k in range(frames[key]['undo']):
                if len(key_log)>0:
                    key_log.pop()
            img = np.zeros((512, 512)).astype('uint8')
            img = render(img,key_log, frames)
        else:
            img = render(img,[key],frames)
            key_log.append(key)

        out = cv2.resize(img,(args.resolution,args.resolution),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.output, '%05d.png' % frameIDs[i]), out)
        # cv2.imwrite(os.path.join(args.output,'%05d.png'%frameIDs[i]),segmap2rgb(out)[...,::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-fps', type=int,default=4)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    main(args)