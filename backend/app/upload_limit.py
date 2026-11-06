"""Bound multipart bodies before parsing and keep accepted uploads in memory."""
from starlette.formparsers import MultiPartParser
from starlette.responses import JSONResponse
from backend.app.preprocessing import MAX_UPLOAD_BYTES

# A 10 MB file plus bounded multipart headers and fields. Requests with larger
# multipart metadata are rejected too, even if the file itself is smaller.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024


class UploadLimitMiddleware:
    def __init__(self, app):
        self.app = app
        # Starlette otherwise rolls UploadFile over to temporary disk at 1 MB.
        # The receive limit below runs before the parser and bounds this spool.
        MultiPartParser.spool_max_size = MAX_REQUEST_BYTES + 1

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/api/predict':
            return await self.app(scope, receive, send)
        rejection = JSONResponse({'detail': 'Upload request exceeds the 10 MB image limit plus 64 KB multipart allowance'},
                                 status_code=413, headers={'Cache-Control': 'no-store'})
        for key, value in scope.get('headers', []):
            if key.lower() == b'content-length':
                try:
                    if int(value) > MAX_REQUEST_BYTES:
                        return await rejection(scope, receive, send)
                except ValueError:
                    return await JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)(scope, receive, send)
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                return await rejection(scope, receive, send)
            chunks.append(chunk)
            if not message.get('more_body', False):
                break
        body = b''.join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()

        return await self.app(scope, bounded_receive, send)
