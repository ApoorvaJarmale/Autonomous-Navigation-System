
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Flatten
import cv2
import numpy as np
import urllib.request
import socket, time



tang = [900, 910, 920, 933, 945, 956, 966, 976, 986, 1000, 1010, 1020, 1030, 1040, 1050, 1058, 1068, 1078, 1090, 1100,
        1110, 1125, 1135, 1145, 1156, 1167, 1177, 1187, 1197, 1210, 1222, 1232, 1242, 1255, 1265, 1275, 1285, 1296,
        1307, 1318, 1330, 1340, 1350, 1360, 1373, 1383, 1393, 1404, 1414, 1424, 1434, 1444, 1456, 1466, 1478, 1488,
        1498, 1508, 1520, 1530, 1543, 1553, 1563, 1573, 1585, 1595, 1605, 1615, 1625, 1638, 1648, 1658, 1670, 1680,
        1691, 1702, 1712, 1722, 1734, 1744, 1754, 1764, 1774, 1788, 1798, 1808, 1818, 1830, 1840, 1850, 1860, 1865,
        1878, 1885, 1893, 1904, 1915, 1927, 1938, 1949, 1956, 1966, 1976, 1984, 1994, 2004, 2014, 2024, 2034, 2044,
        2054, 2064, 2074, 2084, 2094]


def Tcp_connect(HostIp, Port):
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HostIp, Port))
    return


def Tcp_Write(D):
    s.send((D + '\r').encode())
    return


def Tcp_Read():
    a = ' '
    b = ''
    while a != '\r':
        a = s.recv(1)
        a = a.decode()
        b = b + a
    return b


def Tcp_Close():
    s.close()
    return
## Defining variables
pr_threshold = 1
new_size_col = 92
new_size_row = 69
url = 'http://192.168.43.1:8080/shot.jpg'

def preprocess(image):
    # get shape and chop off 1/3 from the top
    shape = image.shape
    # print("shape: " + str(shape))
    # note: numpy arrays are (row, col)!
    # image = image[shape[0]//4:shape[0]-25, 0:shape[1]]
    # image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (new_size_col, new_size_row), interpolation=cv2.INTER_AREA)
    return image


def get_model():
    # model start here
    input_shape = (new_size_row, new_size_col, 3)
    filter_size = 3
    pool_size = (2, 2)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))

    # model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1)

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    # filenames = glob.glob("imgs/*.jpg")
    Tcp_connect('192.168.43.151', 17098)
    model = get_model()
    model.compile("adam", "mse")
    weights_file = 'model_best.h5'
    model.load_weights(weights_file)


    while True:
        imgResp = urllib.request.urlopen(url)

        # Numpy to convert into a array
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

        # Finally decode the array to OpenCV usable format ;)
        img = cv2.imdecode(imgNp, -1)

        # put the image on screen
        cv2.imshow('IPWebcam', img)

        image_array = preprocess(img)
        transformed_image_array = image_array[None, :, :, :]
        steering_angle = 1.0 * float(model.predict(transformed_image_array, batch_size=10))
        # steering_angle = 0.7*float(row[1]) + 0.6*float(steering_angle) +0.15*(0.001)*random.randint(0,100)
        print((steering_angle*58)+58)
        steering=int((steering_angle*58)+58)
        trottle=100
        reverse=0
        Tcp_Write(str(trottle) + "_" + str(tang[steering]) + "_" + str(reverse))


        if cv2.waitKey(1) == 27:
            exit(0)
            Tcp_Close()
