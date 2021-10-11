import sys, argparse
sys.path.append('..')

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
import cv2
from ui.util import number_color, number_object
import qdarkstyle
import numpy as np
import datetime
import pickle,time

from PIL import Image
import os

class ExWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.EX = Ex(args)


class Ex(QWidget, Ui_Form):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self.current_style = 0
        if self.args.load_network:
            import torch
            from sofgan import init_deep_model
            self.styles, self.generator = init_deep_model('../modules/sofgan.pt')
            self.noise = [getattr(self.generator.noises, f'noise_{i}') for i in range(self.generator.num_layers)]

        self.setupUi(self)
        self.show()

        self.modes = 0
        self.alpha = 0.5

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.modes, self)
        self.scene.setSceneRect(0, 0, 512, 512)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignCenter)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.GT_scene = QGraphicsScene()
        self.graphicsView_GT.setScene(self.GT_scene)
        self.graphicsView_GT.setAlignment(Qt.AlignCenter)
        self.graphicsView_GT.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_GT.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)

        self.init_screen()

    def init_screen(self):
        self.image = QPixmap(QSize(512, 512))
        self.image.fill(QColor('#000000'))
        self.mat_img = np.zeros([512, 512], np.uint8)

        self.mat_img_org = self.mat_img.copy()

        self.GT_img_path = None
        GT_img = self.mat_img.copy()
        self.GT_img = Image.fromarray(GT_img)
        self.GT_img = self.GT_img.convert('RGB')

        self.last = time.time()

        self.scene.reset()
        if len(self.scene.items()) > 0:
            self.scene.reset_items()

        self.scene_image_pts = self.scene.addPixmap(self.image)
        self.GT_scene_image_pts = self.GT_scene.addPixmap(self.image)

        self.image = np.zeros([512, 512, 3], np.uint8)
        self.image_raw = self.image.copy()
        self.update_segmap_vis(self.mat_img)

        ###############
        self.recorded_img_names = []

        self.frameLog = {}
        self.starTime = datetime.datetime.now().strftime('%H_%M_%S_%f')

    def run_deep_model(self):
        ""
        if self.args.load_network:
            with torch.no_grad():
                seg_label = torch.from_numpy(self.id_remap(self.mat_img)).view(1,1,512,512).float().cuda()
                fake_img, _, _, _ = self.generator(self.styles[self.current_style%len(self.styles)], return_latents=False,
                                                   condition_img=seg_label, input_is_latent=True, noise=self.noise)
                fake_img = ((fake_img[0].permute(1,2,0).cpu()+ 1) / 2 * 255).clamp_(0,255).numpy().astype('uint8')
                fake_img = cv2.resize(fake_img, (512, 512))
            self.GT_scene_image_pts.setPixmap(QPixmap.fromImage(QImage(fake_img.data.tobytes(), \
                                                             fake_img.shape[1], fake_img.shape[0],
                                                             QImage.Format_RGB888)))
        else:
            print('Did not load the deep model, you need to specify --load_network if you want to render rgb images')

    def change_style(self):
        self.current_style += 1
        self.run_deep_model()

    @pyqtSlot()
    def open(self):

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", 'F:/Lab/samples')
        if fileName:

            self.mat_img_path = os.path.join(fileName)
            self.fileName = fileName

            # USE CV2 read images, because of using gray scale images, no matter the RGB orders
            mat_img = cv2.imread(self.mat_img_path, -1)
            if mat_img is None:
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            if mat_img.ndim == 2:
                self.mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_NEAREST)
                self.image = self.segmap2rgb(self.id_remap(self.mat_img))
                self.mat_img_org = self.mat_img.copy()
            else:
                self.image = cv2.resize(mat_img[...,::-1], (512, 512))

            self.image_raw = self.image.copy()
            self.image = np.round(self.alpha*self.image).astype('uint8')
            image = self.image + (self.segmap2rgb(self.id_remap(self.mat_img)) * int(1000 * (1.0 - self.alpha)) // 1000).astype('uint8')
            image = QPixmap.fromImage(QImage(image.data.tobytes(), self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))


            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()

            self.scene_image_pts = self.scene.addPixmap(image)

            if mat_img.ndim == 2: # template
                self.update_segmap_vis(self.mat_img)

    @pyqtSlot()
    def open_reference(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",QDir.currentPath()+'/samples')
        if fileName:

            self.mat_img_path = os.path.join(fileName)
            self.fileName = fileName

            mat_img = cv2.imread(self.mat_img_path, 1)

            self.image_raw = cv2.resize(mat_img[...,::-1], (512, 512))
            self.change_alpha_value()

    def update_segmap_vis(self, segmap):
        ""

        if not self.args.load_network:
            self.GT_scene_image_pts.setPixmap(QPixmap.fromImage(QImage((10 * segmap).data.tobytes(), \
                                                             segmap.shape[1], segmap.shape[0],
                                                             QImage.Format_Grayscale8)))

        out = self.image + (self.segmap2rgb(self.id_remap(self.mat_img))*int(1000*(1.0-self.alpha))//1000).astype('uint8')
        self.scene_image_pts.setPixmap(QPixmap.fromImage(QImage(out.data.tobytes(), \
                                                         out.shape[1], out.shape[0],
                                                         QImage.Format_RGB888)))

        print('FPS: %s'%(1.0/(time.time()-self.last)))
        self.last = time.time()


    @pyqtSlot()
    def change_brush_size(self):
        self.scene.brush_size = self.brushSlider.value()
        self.brushsizeLabel.setText('Brush size: %d' % self.scene.brush_size)


    @pyqtSlot()
    def change_alpha_value(self):
        self.alpha = self.alphaSlider.value() / 20
        self.alphaLabel.setText('Alpha: %.2f' % self.alpha)

        self.image = np.round(self.image_raw*self.alpha).astype('uint8')
        out = self.image + (self.segmap2rgb(self.id_remap(self.mat_img))*int(1000*(1.0-self.alpha))//1000).astype('uint8')

        self.scene_image_pts.setPixmap(QPixmap.fromImage(QImage(out.data.tobytes(), \
                                                         out.shape[1], out.shape[0],
                                                         QImage.Format_RGB888)))


    @pyqtSlot()
    def mode_select(self, mode):
        self.modes = mode
        self.scene.modes = mode

        if mode == 0:
            self.brushButton.setStyleSheet("background-color: #85adad")
            self.recButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 1:
            self.recButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.fillButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        elif mode == 2:
            self.fillButton.setStyleSheet("background-color: #85adad")
            self.brushButton.setStyleSheet("background-color:")
            self.recButton.setStyleSheet("background-color:")
            QApplication.setOverrideCursor(Qt.PointingHandCursor)

    def segmap2rgb(self, img):
        part_colors = np.array([[0, 0, 0], [127, 212, 255], [255, 255, 127], [255, 255, 170],  # 'skin',1 'eye_brow'2,  'eye'3
                       [240, 157, 240], [255, 212, 255],  # 'r_nose'4, 'l_nose'5
                       [89, 64, 92], [237, 102, 99], [181, 43, 101],  # 'mouth'6, 'u_lip'7,'l_lip'8
                       [0, 255, 85], [0, 255, 170],  # 'ear'9 'ear_r'10
                       [255, 255, 170],
                       [127, 170, 255], [85, 0, 255], [255, 170, 127],  # 'neck'11, 'neck_l'12, 'cloth'13
                       [212, 127, 255], [0, 170, 255],  # , 'hair'14, 'hat'15
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255], [100, 150, 200]]).astype('int')

        condition_img_color = part_colors[img]
        return condition_img_color


    def id_remap(self, seg):
        remap_list = np.array([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).astype('uint8')
        return remap_list[seg.astype('int')]

    @pyqtSlot()
    def save_img(self):

        ui_result_folder = './ui_results/' + os.path.basename(self.fileName)[:-4]

        os.makedirs(ui_result_folder,exist_ok=True)

        outName = os.path.join(ui_result_folder,datetime.datetime.now().strftime('%m%d%H%M%S') + '_segmap.png')
        cv2.imwrite(outName, self.mat_img)
        print('===> save segmap to %s'%outName)


    @pyqtSlot()
    def switch_labels(self, label):
        self.scene.label = label
        self.scene.color = number_color[label]
        _translate = QCoreApplication.translate
        self.color_Button.setText(_translate("Form", "%s" % number_object[label]))
        self.color_Button.setStyleSheet("background-color: %s;" % self.scene.color+ " color: black")


    @pyqtSlot()
    def undo(self):
        self.scene.undo()

    def startScreening(self):
        self.isScreening,self.frameLog = True,{}
        self.starTime = datetime.datetime.now().strftime('%H_%M_%S_%f')

    def saveScreening(self):
        os.makedirs('./frameLog', exist_ok=True)
        name = './frameLog/%s.pkl'%self.starTime
        with open(name, 'wb') as f:
            pickle.dump(self.frameLog, f)
        print('====> saved frame log to %s'%name)

    def cleanForground(self):
        self.mat_img[:] = 0
        self.update_segmap_vis(self.mat_img)
        self.frameLog[datetime.datetime.now().strftime('%H:%M:%S:%f')] = {'undo': len(self.frameLog.keys())}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--load_network', default=False, action="store_true",
                        help='load network')
    # parser.add_argument('-i', '--input', type=str)
    # parser.add_argument('-o', '--output', type=str)
    # parser.add_argument('-fps', type=int,default=4)
    # parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = ExWindow(args)
    sys.exit(app.exec_())
