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
        cursor.execute("SELECT contents FROM scrap_object WHERE hash=:hash", {"hash": hash})
        row = cursor.fetchone()
        if row:
            flat = row[0]
            response.status = 200
            response.content_type = "application/scrapscript"
            response.content_length = len(flat)
            return flat
        else:
            response.status = 404
            return "Not found.\n"


@post("/scrap/upload")
def upload_scrap():
    # curl -X POST -F smallstr=@smallstr.flat -F foo=bar -F a=b  ...
    # for key, value in request.forms.items():
    #     print(key, value)
    inserted = set()
    items = []
    for key, value in request.files.items():
        flat = value.file.read()
        hash = hashlib.sha256(flat).hexdigest()
        items.append({"hash": hash, "contents": flat})
    with db:
        cursor = db.cursor()
        result = cursor.executemany("INSERT or IGNORE INTO scrap_object VALUES (:hash, :contents)", items)
        num_inserted = result.rowcount
    if num_inserted > 0:
        response.status = 201
        return {"num_inserted": num_inserted}
    response.status = 304
    return {"num_inserted": 0}


if IN_MEMORY:
    @post("/persist")
    def persist_db():
        disk_db = sqlite3.connect("land.db")
        db.backup(disk_db)
        return "Flushed to disk.\n"


if __name__ == "__main__":
    run(host="localhost", port=8080, reloader=True)
