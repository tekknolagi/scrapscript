import hashlib
import sqlite3
from bottle import get, post, run, template, request


# db = sqlite3.connect("land.db")
db = sqlite3.connect(":memory:")
with db:
    cursor = db.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS scrap_object (hash TEXT, contents BLOB)")
    cursor.execute("CREATE TABLE IF NOT EXISTS scrap_source (hash TEXT, contents TEXT)")


@get("/scrap/sha256/<name>")
def find_scrap(name):
    return template("<b>Hello {{name}}</b>!", name=name)

@post("/scrap/upload")
def upload_scrap():
    # curl -X POST -F smallstr=@smallstr.flat -F foo=bar -F a=b  ...
    for key, value in request.forms.items():
        print(key, value)
    for key, value in request.files.items():
        print(key, value.file.read())
    return "<b>Hello and thank you</b>!"


if __name__ == "__main__":
    run(host="localhost", port=8080)
