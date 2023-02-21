
def image_grid(array, ncols=8):
        index, height, width, channels = array.shape
        nrows = index//ncols

        img_grid = (array.reshape(nrows, ncols, height, width)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols))

        return img_grid