
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
from datetime import datetime


class ReferenceDialog(QDialog):
    def __init__(self, Parent):
        QDialog.__init__(self, Parent)
        self.Form = Parent


class SnapshotDialog(QDialog):
    def __init__(self, Parent):
        QDialog.__init__(self, Parent)
        self.Form = Parent
        self.count = 0

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_Left:
            if self.count -1 >= 0:
                self.Form.open_snapshot_dialog(self.count -1)
        elif e.key() == Qt.Key_Right:
            if self.count + 1 < 15:
                self.Form.open_snapshot_dialog(self.count +1)


class GraphicsScene(QGraphicsScene):
    def __init__(self, modes, Form):
        QGraphicsScene.__init__(self)
        self.modes = modes
        self.mouse_clicked = False
        self.prev_pt = None
        self.history_list = []

        # brush color
        self.color = '#7fd4ff'
        self.label = 1
        self.brush_size = 6
        self.Form = Form


    def reset(self):
        self.prev_pt = None

        self.history_list = []


    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)



    def mousePressEvent(self, event):
        self.mouse_clicked = True

        if self.modes == 1:
            self.rec_top_left = event.scenePos()
            self.old_recItem = None

        elif self.modes == 2:

            img_current_point = (int(event.scenePos().y()), int(event.scenePos().x()))
            scene_current_point= (int(event.scenePos().x()), int(event.scenePos().y()))

            current_color_label = self.Form.mat_img[img_current_point]
            thresh = np.uint8(self.Form.mat_img == current_color_label) * 255
            cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            Contours_num = None

            for i in range(len(cnts[0])):
                whether_in_shape = cv2.pointPolygonTest(cnts[0][i], scene_current_point, False)
                if whether_in_shape == 1:
                    Contours_num = i
                    break

            if Contours_num != None:
                # qpoints = [QPointF(pt[0][0], pt[0,1]) for pt in cnts[0][Contours_num]]
                # PolygonItem = QGraphicsPolygonItem(QPolygonF(qpoints))
                # PolygonItem.setBrush(QBrush(QColor(self.Form.scene.color)))
                # PolygonItem.setPen(QPen(QColor(self.Form.scene.color), 2, Qt.SolidLine))
                #
                # self.addItem(PolygonItem)


                fill = {}
                fill['contours'] = cnts[0]
                fill['contours_num'] = Contours_num
                fill['label'] = self.label
                fill['shape'] = 'Fill'

                self.history_list.append(fill)
                self.Form.frameLog[datetime.now().strftime('%H:%M:%S:%f')] = fill
                self.convert_fill(fill)
                self.Form.update_segmap_vis(self.Form.mat_img)


    def mouseReleaseEvent(self, event):
        self.prev_pt = None
        self.mouse_clicked = False

        if self.modes == 1:
            self.old_recItem = None



    def mouseMoveEvent(self, event):
        if self.mouse_clicked:
            # print(event.scenePos())
            if self.modes == 0:
                if self.prev_pt:
                    self.drawStroke(self.prev_pt, event.scenePos())
                    self.prev_pt = event.scenePos()

                else:
                    self.prev_pt = event.scenePos()

            if self.modes == 1:
                self.drawRec(self.rec_top_left, event.scenePos())


    def drawStroke(self, prev_pt, curr_pt):
        # lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        # lineItem.setPen(QPen(QColor(self.color), self.brush_size, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))  # rect
        # self.addItem(lineItem)

        stroke = {}
        stroke['prev'] = (int(prev_pt.x()), int(prev_pt.y()))
        stroke['curr'] = (int(curr_pt.x()), int(curr_pt.y()))
        stroke['label'] = self.label
        stroke['brush_size'] = self.brush_size
        stroke['shape'] = 'Stroke'
        self.history_list.append(stroke)
        self.Form.frameLog[datetime.now().strftime('%H:%M:%S:%f')] = stroke
        self.convert_stroke(stroke)
        self.Form.update_segmap_vis(self.Form.mat_img)

    def drawRec(self, prev_pt, curr_pt):

        top_left =  (int(min(prev_pt.x(), curr_pt.x())), int(min(prev_pt.y(), curr_pt.y())))
        bottom_right = (int(max(prev_pt.x(), curr_pt.x())), int(max(prev_pt.y(), curr_pt.y())))

        # recItem = QGraphicsRectItem(QRectF(QPointF(top_left[0], top_left[1]), QPointF(bottom_right[0], bottom_right[1])))
        # recItem.setBrush(QBrush(QColor(self.Form.scene.color)))
        # recItem.setPen(QPen(Qt.NoPen))
        #
        # self.addItem(recItem)

        if self.old_recItem == None:
            # self.old_recItem = recItem
            self.old_rec_mat_img = self.Form.mat_img.copy()
        else:
            # self.removeItem(self.old_recItem)
            # self.old_recItem = recItem
            self.history_list.pop()

        rec = {}


        rec['prev'] = top_left
        rec['curr'] = bottom_right

        rec['label'] = self.label
        rec['brush_size'] = None
        rec['shape'] = 'Rec'

        self.history_list.append(rec)
        self.Form.frameLog[datetime.now().strftime('%H:%M:%S:%f')] = rec

        self.Form.mat_img = self.old_rec_mat_img.copy()
        self.convert_rec(rec)
        self.Form.update_segmap_vis(self.Form.mat_img)

    def convert_stroke(self, stroke_point):
        if len(stroke_point) == 5:
            color = stroke_point['label']
            cv2.line(self.Form.mat_img, stroke_point['prev'], stroke_point['curr'], color, stroke_point['brush_size'])
        else:
            print("wrong stroke")


    def convert_rec(self, rectangle):
        if len(rectangle) == 5:
            color = rectangle['label']
            cv2.rectangle(self.Form.mat_img, rectangle['prev'], rectangle['curr'], color, -1)
        else:
            print("wrong rectangle")

    def convert_fill(self, fill):
        if len(fill) == 4:
            color = fill['label']
            cv2.drawContours(self.Form.mat_img, fill['contours'], fill['contours_num'],color, -1)
        else:
            print("wrong fill")




    def undo(self):
        steps = 1
        if len(self.history_list)>1:
            print(self.history_list[-1]['shape'])
            if self.history_list[-1]['shape'] == 'Rec':
                # item = self.items()[0]
                # self.removeItem(item)
                self.history_list.pop()


            elif self.history_list[-1]['shape'] == 'Stroke':
                if len(self.history_list)>=6:
                    steps = 6
                    for i in range(steps):
                        # item = self.items()[0]
                        # self.removeItem(item)
                        self.history_list.pop()
                else:
                    steps = len(self.history_list)-1
                    for i in range(len(self.history_list)-1):
                        # item = self.items()[0]
                        # self.removeItem(item)
                        self.history_list.pop()
            elif self.history_list[-1]['shape'] == 'Fill':
                # item = self.items()[0]
                # self.removeItem(item)
                self.history_list.pop()

        self.Form.mat_img = self.Form.mat_img_org.copy()
        for pts in self.history_list:
            if pts['shape'] == 'Stroke':
                self.convert_stroke(pts)
            elif pts['shape'] == 'Rec':
                self.convert_rec(pts)
            elif pts['shape'] == 'Fill':
                self.convert_fill(pts)

        self.Form.update_segmap_vis(self.Form.mat_img)
        self.Form.frameLog[datetime.now().strftime('%H:%M:%S:%f')]= {'undo':steps}




