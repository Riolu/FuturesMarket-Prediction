import os
import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing

#Given (t-30,t), predict the value on t+5
STEP = 5
period = 30

# filenames = []
# DBExport = 'C:\Users\DELL-PC\PycharmProjects\Market-Prediction\DBExport'
# for filename in os.listdir(DBExport):
#     if filename.endswith(".csv"):
#         filenames.append(os.path.join(DBExport, filename))
# filenames.sort()

filenames = ["TK_rb0000[s20160101 00000000_e20160110 00153000]20170410_1722_0.csv"]

labels = []
total = []


def get_label(file_id):
    global STEP, labels   # STEP=20 for 20 ticks
    filename = filenames[file_id]   # the filename of the csv file you want to process
    df = pd.read_csv(filename)    # a list of tuple (id,0/1)
    #df = pd.read_csv("test.csv")
    count = len(df)
    print(count)


    for i in range(period, count, 1):
        features = []
        if(i+STEP>=count): break
        # to keep efficiency, if STEP is smaller than the last k, we just set the same label

        cur_ask_bid = df.iloc[i]['AskPrice1'] + df.iloc[i]['BidPrice1']
        next_ask_bid = df.iloc[i+STEP]['AskPrice1'] + df.iloc[i+STEP]['BidPrice1']

        # if price(t) == price (t + STEP), we are going to check price(t+STEP+k), k starts with 1
        k = 1
        while next_ask_bid == cur_ask_bid:
            if i+STEP+k >= count: # out of range
                next_ask_bid = df.iloc[-1]['AskPrice1'] + df.iloc[-1]['BidPrice1']
                break
            next_ask_bid = df.iloc[i+STEP+k]['AskPrice1'] + df.iloc[i+STEP+k]['BidPrice1']
            k += 1
        # here, we know [t+STEP,t+STEP+k] have the same price, so we can use this information to avoid wasteful computation

        if cur_ask_bid < next_ask_bid:
            labels.append(1)# 1 means future -> increase
        elif cur_ask_bid > next_ask_bid:
            labels.append(0)  # 0 means future -> decrease


        for j in range(period):
            cnn = []
            cnn.append(df.iloc[i-period+j+1]['AskPrice1'] + df.iloc[i-period+j+1]['BidPrice1'])
            cnn.append(df.iloc[i-period+j+1]['AskPrice1'])
            cnn.append(df.iloc[i-period+j+1]['BidPrice1'])
            cnn.append(df.iloc[i-period+j+1]['AskVolume1'])
            cnn.append(df.iloc[i-period+j+1]['BidVolume1'])
            features.append(cnn)

        features = np.array(features)
        features = features.reshape((1,5*period))[0]
        #print(features)
        total.append(preprocessing.scale(features))

        if (len(features)!=len(labels)): #I found that len(X)!=len(y) finally
            break

        if (i % int(0.01*count)==0):
              print (i/int(0.01*count))

    X = []
    y = []

    X = np.array(total)
    y = np.array(labels)
    print (len(y))

    print("Start Saving")
    np.save("data/data_X.npy", X)
    np.save("data/data_y.npy", y)
    print("Finish Saving")





    # # save to a txt file
    # output = open("data2.pkl", 'wb')
    # out = np.array(total)
    # pickle.dump(out, output)
    # output.close()
    # output_label = open("label2.pkl", 'wb')
    # pickle.dump(np.array(labels), output_label)
    # #f2.close()
    # print(out.shape)
    # output_label.close()


ind = 0 # for the first csv file
get_label(ind)



