from lxml import etree
from StackOverflowSqlite import StackOverflowSqlite
import time
import os
BASE_DIR = "data"

posts_xml_file = open(os.path.join(BASE_DIR, "Posts.xml"), "rb")
etree.XMLParser(recover=True)


DB = os.path.join(BASE_DIR, "stackoverflow-posts.db")
#SQLITE = StackOverflowSqlite(DB)

def main():
    posts_count = 0
    end_tags_count = 0
    maxcount = 10
    processed_count = 0
    #524409
    REQUIRED_TAG = "row"
    for event, elem in etree.iterparse(posts_xml_file, events=("start", "end")):
        if (event == "end" and elem.tag == REQUIRED_TAG):
            if elem.get("PostTypeId") == "1":
                if posts_count > processed_count:
                    getElementDataAndStore(elem)
                posts_count+=1
                if posts_count % 1000 == 0:
                        print(str(posts_count) + "::" + str(end_tags_count))
            #time.sleep(0.01)
        end_tags_count+=1
    print("Total tags processed " + str(end_tags_count))
    print("Total posts processed " + str(posts_count))
    posts_xml_file.close()
    #SQLITE.close()

def getElementDataAndStore(elem):
    id = elem.get("Id")
    postObject = {}
    postObject["id"] = elem.get("Id")
    postObject["tags"] =  elem.get("Tags").encode('utf8')
    postObject["title"]  = elem.get("Title").encode('utf8')
    postObject["content"] = elem.get("Body").encode('utf8')
    print(postObject)
    #SQLITE.commit(postObject)  
    time.sleep(0.05)

if __name__ == "__main__":
    main()