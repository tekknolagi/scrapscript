import hashlib
import sqlite3
from bottle import get, post, run, template, request, response

IN_MEMORY = False

if IN_MEMORY:
    db = sqlite3.connect(":memory:")
else:
    db = sqlite3.connect("land.db")
with db:
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS scrap_object (hash TEXT, contents BLOB)")
    cursor.execute("CREATE TABLE IF NOT EXISTS scrap_source (hash TEXT, contents TEXT)")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS scrap_contents on scrap_object ( contents )")


@get("/scrap/sha256/<hash>")
def find_scrap(hash):
    with db:
        cursor = db.cursor()
        cursor.execute("SELECT contents FROM scrap_object WHERE hash=?", (hash,))
        row = cursor.fetchone()
        if row:
            flat = row[0]
            response.content_type = "application/scrapscript"
            response.content_length = len(flat)
            return flat
        else:
            return "Not found."

@post("/scrap/upload")
def upload_scrap():
    # curl -X POST -F smallstr=@smallstr.flat -F foo=bar -F a=b  ...
    # for key, value in request.forms.items():
    #     print(key, value)
    with db:
        cursor = db.cursor()
        for key, value in request.files.items():
            flat = value.file.read()
            hash = hashlib.sha256(flat).hexdigest()
            cursor.execute("INSERT or IGNORE INTO scrap_object VALUES (?, ?)", (hash, flat))
    return template("Inserted into DB as {{hash}}", hash=hash)


if IN_MEMORY:
    @post("/persist")
    def persist_db():
        disk_db = sqlite3.connect("land.db")
        db.backup(disk_db)
        return "Flushed to disk."


if __name__ == "__main__":
    run(host="localhost", port=8080, reloader=True)
