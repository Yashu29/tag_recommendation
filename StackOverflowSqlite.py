import sqlite3
import time


class StackOverflowSqlite(object):

	def __init__(self,db):
		self.conn = sqlite3.connect(db)
		self.cursor = self.conn.cursor()
		create_table = "create table if not exists posts(id, tags, title, content)"
		self.cursor.execute(create_table)
		self.commit_count = 0
	
	def commit(self, postObject):
		insert_post = "insert into posts values(?, ?, ?, ?)"
		self.cursor.execute(insert_post, (postObject["id"], 
			postObject["tags"], postObject["title"], postObject["content"]))
		self.commit_count += 1
		if (self.commit_count > 1000):
			print("...")
			self.conn.commit()
			self.commit_count = 0
			time.sleep(1)


	def close(self):
		self.conn.commit()
		self.conn.close()

	def getEntryCount():
		return 0