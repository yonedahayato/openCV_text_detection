import cv2 as cv
import numpy as np
import os
import sys

class Text_Detection:
    def __init__(self):
        self.images_path_list = ["image"]
        self.output_dir = "./output"
        self.output_file_name = "sample"
        self.setting()

    def setting(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def read_image(self, image_path):
        for image_path_tmp in self.images_path_list:
            self.img = cv.imread(image_path)
            self.vis = self.img.copy()

    def extract_channels(self):
        self.channels = cv.text.computeNMChannels(self.img)
        cn = len(self.channels) - 1
        for c in range(0, cn):
            self.channels.append((255-self.channels[c]))

    def apply_classifier(self):
        for channel in self.channels:

            erc1 = cv.text.loadClassifierNM1("./trained_classifierNM1.xml")
            er1 = cv.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

            erc2 = cv.text.loadClassifierNM2("./trained_classifierNM2.xml")
            er2 = cv.text.createERFilterNM2(erc2, 0.5)

            regions = cv.text.detectRegions(channel, er1, er2)

            rects = cv.text.erGrouping(self.img, channel, [r.tolist() for r in regions])

            for r in range(0, np.shape(rects)[0]):
                rect = rects[r]
                cv.rectangle(self.vis, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 0), 2)
                cv.rectangle(self.vis, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (225, 225, 225), 1)

        cv.imwrite("{}/{}.png".format(self.output_dir, self.output_file_name), self.vis)

    def detection(self):
        pass

def main():
    td = Text_Detection()
    td.read_image("./sample_2018-05-14 15.56.57.png")
    td.extract_channels()
    td.apply_classifier()

if __name__ == "__main__":
    main()
