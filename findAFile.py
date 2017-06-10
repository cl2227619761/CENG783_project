import os
import shutil

allframes=[]
for set in range(1,4):
    for root, dirs, files in os.walk("D:\\DATASET\\MARIS\\sets\set"+str(set)+"Right\\"):
        for file in files:
            #print(file)
            if file.endswith(".png"):
                fileNameWithPath = os.path.join(root, file)
                leftorright = fileNameWithPath.split('_')[0]
                frameName = leftorright.split('\\')[-1]

                frameNamewithset=frameName+"_set"+str(set)
                os.rename("D:\\DATASET\\MARIS\\sets\set" + str(set) + "Right\\"+frameName+"_r.png","D:\\DATASET\\MARIS\\sets\set" + str(set) + "Left\\"+frameNamewithset+"_1.png")
                # fileNameWithPath=os.path.join(root, file)
                # print(os.path.join(root, file))
                # leftorright=fileNameWithPath.split('_')[1]
                # if leftorright[0]=='r':
                #     shutil.move(fileNameWithPath, "D:\\DATASET\\MARIS\\sets\set"+str(set)+"Right")
                # print("Tasinan Dosya:"+fileNameWithPath)
                # # os.mkdir(folderNameWithPath)
                # # shutil.move(fileNameWithPath,folderNameWithPath)