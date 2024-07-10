# Modified from https://pymotw.com/3/urllib.request/#uploading-files
# See also https://stackoverflow.com/a/681182/569183
import urllib.request
import io
import uuid


class MultiPartForm:
    def __init__(self):
        self.form_fields = []
        self.files = []
        # Use a large random byte string to separate
        # parts of the MIME data.
        self.boundary = uuid.uuid4().hex.encode("utf-8")
        return

    def get_content_type(self):
        boundary = self.boundary.decode("utf-8")
        return f"multipart/form-data; boundary={boundary}"

    def add_file(self, fieldname, filename, fileHandle, mimetype=None):
        body = fileHandle.read()
        if mimetype is None:
            mimetype = "application/octet-stream"
        self.files.append((fieldname, filename, mimetype, body))
        return

    @staticmethod
    def _form_data(name):
        return (f'Content-Disposition: form-data; name="{name}"\r\n').encode("utf-8")

    @staticmethod
    def _attached_file(name, filename):
        return (f'Content-Disposition: file; name="{name}"; filename="{filename}"\r\n').encode("utf-8")

    @staticmethod
    def _content_type(ct):
        return f"Content-Type: {ct}\r\n".encode("utf-8")

    def __bytes__(self):
        buffer = io.BytesIO()
        boundary = b"--" + self.boundary + b"\r\n"

        # Add the form fields
        for name, value in self.form_fields:
            buffer.write(boundary)
            buffer.write(self._form_data(name))
            buffer.write(b"\r\n")
            buffer.write(value.encode("utf-8"))
            buffer.write(b"\r\n")

        # Add the files to upload
        for f_name, filename, f_content_type, body in self.files:
            buffer.write(boundary)
            buffer.write(self._attached_file(f_name, filename))
            buffer.write(self._content_type(f_content_type))
            buffer.write(b"\r\n")
            buffer.write(body)
            buffer.write(b"\r\n")

        buffer.write(b"--" + self.boundary + b"--\r\n")
        return buffer.getvalue()
