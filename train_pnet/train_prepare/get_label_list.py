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

 
def ReadFileDatas(size):
    img_size = size
    FileNamelist = []
    count = 0
    file = open('%s/neg_%s.txt.bak'%(img_size,img_size),'r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 570000):
            break;
    print('neg data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()

    count  = 0
    file = open('%s/pos_%s.txt.bak'%(img_size,img_size),'r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 190000):
            break;
    print('pos data len ( FileNamelist ) = ' ,len(FileNamelist))
    file.close()

    count = 0
    file = open('%s/part_%s.txt.bak'%(img_size,img_size),'r+')
    for line in file:
        line=line.strip('\n') #删除每一行的\n
        FileNamelist.append(line)
        count += 1
        if(count >= 190000):
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
    img_size = 12
    filelist = '%s/neg_%s.txt'%(img_size,img_size)
    filelist_bak = '%s/neg_%s.txt.bak'%(img_size,img_size)
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak) 

    filelist = '%s/pos_%s.txt'%(img_size,img_size)
    filelist_bak = '%s/pos_%s.txt.bak'%(img_size,img_size)
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak)

    filelist = '%s/part_%s.txt'%(img_size,img_size)
    filelist_bak = '%s/part_%s.txt.bak'%(img_size,img_size)
    listFileinfo = ReadSampleDatas(filelist)
    random.shuffle(listFileinfo)
    WriteDatasToSampleList(listFileinfo,filelist_bak)

    listFileInfo = ReadFileDatas(img_size)
    #打乱列表中的顺序
    random.shuffle(listFileInfo)
    WriteDatasToFile(listFileInfo)

