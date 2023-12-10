import pathlib

import pygit2


yard = "/tmp/scrapyard"
repo = pygit2.init_repository(yard, bare=True)

root = repo.TreeBuilder()
obj_id = repo.create_blob(b"hello world from pygit2!")
root.insert("test", obj_id, pygit2.GIT_FILEMODE_BLOB)
tree_id = root.write()

ref = "HEAD"
author = pygit2.Signature("Alice Author", "alice@authors.tld")
message = "Initial commit"
parents: object = []
repo.create_commit(ref, author, author, message, tree_id, parents)

root = repo.TreeBuilder()
obj_id = repo.create_blob(b"goodbye world from pygit2!")
other_obj_id = repo.create_blob(b"hello chris")
root.insert("test", obj_id, pygit2.GIT_FILEMODE_BLOB)
root.insert("another_one", other_obj_id, pygit2.GIT_FILEMODE_BLOB)
tree_id = root.write()

ref = repo.head.name
author = pygit2.Signature("Alice Author", "alice@authors.tld")
message = "Second commit"
parents = [repo.head.target]
repo.create_commit(ref, author, author, message, tree_id, parents)
