#!/usr/bin/python3
import numpy as np

class MaxPool2:
    # Pool of size 2x2

    def get_clusters(self, image):
        # Gets 2x2 areas of an image which will be pooled
        h, w, _ = image.shape
        hh = h // 2
        ww = w // 2

        for i in range(hh):
            for j in range(ww):
                image_cluster = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield image_cluster, i, j

    def forward(self, input):
        h, w, no_filters = input.shape
        output = np.zeros((h // 2, w // 2, no_filters))
        for image_cluster, i, j in self.get_clusters(input):
            output[i, j] = np.amax(image_cluster, axis=(0,1))
        return output
