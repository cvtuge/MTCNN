import random

def ReadSampleDatas(filelist):
    FileNamelist = []
    file = open(filelist,'r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
    print('neg data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()
    return FileNamelist

def WriteDatasToSampleList(listInfo,filelist):
    file_handle=open(filelist,mode='w')
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        file_handle.write(str+'\n')
    file_handle.close()

 
def ReadFileDatas():
    FileNamelist = []
    count = 0
    file = open('48/neg_48.txt.bak','r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 480000):
            break;
    print('neg data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()

    count  = 0
    file = open('48/pos_48.txt.bak','r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 160000):
            break;
    print('pos data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()

    count = 0
    file = open('48/part_48.txt.bak','r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 160000):
            break;
    print('part data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()

    return FileNamelist
 
def WriteDatasToFile(listInfo):
    file_handle=open('label_list.txt',mode='w')
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        file_handle.write(str+'\n')
    file_handle.close()
 
if __name__ == "__main__":
    filelist = '48/neg_48.txt'
    filelist_bak = '48/neg_48.txt.bak'
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak) 

    filelist = '48/pos_48.txt'
    filelist_bak = '48/pos_48.txt.bak'
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak)

    filelist = '48/part_48.txt'
    filelist_bak = '48/part_48.txt.bak'
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak)

    listFileInfo = ReadFileDatas()
    #打乱列表中的顺序
    random.shuffle(listFileInfo)
    WriteDatasToFile(listFileInfo)

