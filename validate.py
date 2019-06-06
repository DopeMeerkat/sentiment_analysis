# -*- coding: UTF-8 -*-

import re
import sys
from io import open
def parse(filename):
    sentimentList = []
    pat = re.compile(r'[\u4e00-\u9fa6]+')
    print('reading ' + filename)
    with open(filename, 'r', encoding="UTF-8") as f:
        for line in f:
            arr = line.split(' ', 9)
            total = arr[0].split('Total:')[1]
            sentiments = []
            for i in range(1, 8):
                sentiments.append(int(arr[i].split(':')[1]))
            sentiments.append(int(arr[8].split(':')[1].split('\t')[0]))
            sentiments1 = sentiments.index(max(sentiments))
            sentimentList.append(sentiments1)
    return sentimentList

if __name__ == '__main__':
    sentimentList = parse('sina/sinanews.train')  
    class_names = {"感动":0, "同情":1, "无聊":2, "愤怒":3, "搞笑":4 ,"难过":5,"新奇":6,"温馨":7}
    correct = 0
    total = 0
    with open(sys.argv[1], 'r', encoding="UTF-8") as f:
        for line in f:
            # print(total)
            # print(class_names[line.split('\t')[1].split('\n')[0]])
            # print(sentimentList[total])
            if class_names[line.split('\t')[1].split('\n')[0]] == sentimentList[total]:
                correct += 1
            total += 1
    print('total   \t' + str(total))
    print('correct \t' + str(correct))
    print('accuracy\t' + str(correct/total))
