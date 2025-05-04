import numpy as np

class Conv_3x3:
    # Convolutional Layer which interates over an array and apply a set of 3x3 filters
    def __init__(self, no_filters):
        # Setup the filters

        self.no_filters = no_filters

        # Dividing by 9 means the variance isn't too large
        self.filters = np.random.randn(no_filters, 3, 3) / 9

    def get_clusters(self, image):
        # Get arrays of 3x3 sections of the image for filtering

        w, h = image.shape

        # Subtract 2 from the height and width as we are using 'valid' padding
        for i in range(h - 2):
            for j in range(w - 2):
                image_cluster = image[i:i+3, j:j+3]
                yield image_cluster, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.no_filters))

        for image_cluster, i, j in self.get_clusters(input):
            output[i, j] = np.sum(image_cluster * self.filters, axis=(1, 2))

        return output

