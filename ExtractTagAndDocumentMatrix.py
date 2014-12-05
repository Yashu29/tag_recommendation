from TextsDAO import TextsDAO
import numpy
import json
import operator
import os
BASE_DIR = "data"

DB = os.path.join(BASE_DIR,"stackoverflow-posts.db")
DOC_TO_TAG =  os.path.join(BASE_DIR, "doc_to_tag.npy")
TAG_MAP = os.path.join(BASE_DIR, "tag_to_id.dict")
TID_TO_TAG = os.path.join(BASE_DIR, "tid_to_tag.npy")

TAG_TO_ID_DICT = {}
PID_TO_TAGS_LIST = []
ID_TO_TAG_LIST = []
TAG_COUNT = 0

TYPE = "load"


def main():
    global PID_TO_TAGS_LIST
    global TAG_TO_ID_DICT
    global ID_TO_TAG_LIST
    if TYPE == "load":
        PID_TO_TAGS_LIST = numpy.load(DOC_TO_TAG).tolist()
        with open(TAG_MAP, "r") as f:
            TAG_TO_ID_DICT = json.loads(f.read())
        sorted_tag_to_id_dict = sorted(TAG_TO_ID_DICT.items(), key=operator.itemgetter(1))
        for tag, tid in sorted_tag_to_id_dict:
            print("{} -- {}".format(tid, tag))
            ID_TO_TAG_LIST.append(tag)
        nparray = numpy.array(ID_TO_TAG_LIST)
        numpy.save(TID_TO_TAG, nparray)
        print("{}".format(len(PID_TO_TAGS_LIST)))
        print("{}".format(len(TAG_TO_ID_DICT)))
        return

    ####*******SAVE********###
    texts = TextsDAO(BASE_META_DIR, DB, get_tags=True)
    count = 0
    for id,tags in texts:
        #print("For {} Tags{}".format(id, tags))
        tag_list = []
        for tag in tags:
            tag_list.append(findInDictionary(tag))
        PID_TO_TAGS_LIST.append(tag_list)    
        count += 1
    print("{}".format(PID_TO_TAGS_LIST))
    print("{}".format(TAG_TO_ID_DICT))
    nparray = numpy.array(PID_TO_TAGS_LIST)
    numpy.save(DOC_TO_TAG, nparray)
    with open(TAG_MAP, "w") as f:
        f.write(json.dumps(TAG_TO_ID_DICT))
    texts.close()

def findInDictionary(tag):
    global TAG_COUNT
    if tag in TAG_TO_ID_DICT:
        return TAG_TO_ID_DICT.get(tag)
    else:
        TAG_TO_ID_DICT[tag] = TAG_COUNT
        print("Inserting tag " + tag + " with id " + str(TAG_COUNT))
        TAG_COUNT+=1
        return TAG_TO_ID_DICT[tag]


if __name__ == "__main__":
    main()