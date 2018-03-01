import cv2
import numpy as np
import scipy.cluster
import urllib.request
import socket
from ssl import SSLError
from socket import error as SocketError

NUM_CLUSTERS = 10 # numver of cluster, bigger this number, more accurate the prevalent colors

urls_file = 'urls.txt'
outputCSV = 'prevalent_colors.csv'

f_out = open(outputCSV, "w")

with open(urls_file) as f:
    while True:
        row = f.readline()

        if not row:
            break   # end of file

        url = row.rstrip()

        # read image
        try_times = 0
        while try_times < 3:   # try up to 3 times
            try:
                resp = urllib.request.urlopen(url, timeout=50)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                break
            except urllib.error.URLError as e:
                print(type(e))
                try_times +=1
                continue
            except socket.timeout as e:
                print(type(e))
                try_times +=1
                continue
            except SSLError as e:
                print(type(e))
                try_times +=1
                continue
            except SocketError as e:
                print(type(e))
                try_times +=1
                continue

        im = cv2.imdecode(image, cv2.IMREAD_COLOR)
        im1 = cv2.resize(im, (int(100*im.shape[1]/im.shape[0]), 100) )  # resize the image to smaller size to reduce precessing time
        shape = im1.shape
        ar = im1.reshape(np.prod(shape[:2]), shape[2]).astype(float)

        # finding clusters
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        #print ('cluster centres:\n', codes)

        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

        if counts.size > 3:
            ind = np.argpartition(counts, -3)[-3:]     # find top three color, not sorted
        else: # if number of colors in image is less then 3, output whatever
            ind = np.arange(0, counts.size)

        line = url

        for i in range(ind.size):
            peak = codes[ind[i],:]
            peak1 = [format(int(c), 'X') for c in peak]
            color = "".join(peak1)
            line = line + ',''#'+ color

        line += '\n'
        print (line)
        f_out.write(line)
f_out.close()
