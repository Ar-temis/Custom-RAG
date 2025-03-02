import os

def files(path):
    for filename in os.listdir(path):
        # TODO: add a logging mechanism
        # with open(os.path.join(path, ".FilesAdded.txt"), "wr")
        file_path = path + filename
        # print(file_path)
        print(os.path.join(path, filename))

files("./DataFiles")