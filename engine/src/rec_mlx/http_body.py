"""Bounded HTTP/1 request-body framing, before handing JSON to mlx-lm."""

MAX_BODY = 8 * 1024 * 1024
MAX_LINE = 8192
MAX_OVERHEAD = 64 * 1024


def read_body(stream, headers):
    lengths = headers.get_all("Content-Length", [])
    encodings = headers.get_all("Transfer-Encoding", [])
    if encodings:
        if lengths or len(encodings) != 1 or encodings[0].strip().lower() != "chunked":
            raise ValueError("Use either Content-Length or Transfer-Encoding: chunked")
        body = bytearray()
        overhead = 0

        def line():
            nonlocal overhead
            value = stream.readline(MAX_LINE + 1)
            overhead += len(value)
            if len(value) > MAX_LINE or overhead > MAX_OVERHEAD or not value.endswith(b"\r\n"):
                raise ValueError("Invalid or excessive chunk framing")
            return value[:-2]

        while True:
            size_text = line().split(b";", 1)[0]
            if not size_text or any(c not in b"0123456789abcdefABCDEF" for c in size_text):
                raise ValueError("Invalid chunk size")
            size = int(size_text, 16)
            if len(body) + size > MAX_BODY:
                raise ValueError("Request body must be between 1 byte and 8 MiB")
            if size == 0:
                # Trailers are deliberately discarded, never promoted into headers.
                while trailer := line():
                    if b":" not in trailer or trailer[:1] in (b" ", b"\t"):
                        raise ValueError("Invalid chunk trailer")
                break
            chunk = stream.read(size)
            if len(chunk) != size or stream.read(2) != b"\r\n":
                raise ValueError("Truncated or invalid chunk")
            overhead += 2
            body.extend(chunk)
        if not body:
            raise ValueError("Request body must be between 1 byte and 8 MiB")
        return bytes(body)
    length_text = lengths[0].strip() if len(lengths) == 1 else ""
    if not length_text.isascii() or not length_text.isdecimal():
        raise ValueError("A single Content-Length or Transfer-Encoding: chunked is required")
    length = int(length_text)
    if not 0 < length <= MAX_BODY:
        raise ValueError("Request body must be between 1 byte and 8 MiB")
    body = stream.read(length)
    if len(body) != length:
        raise ValueError("Truncated request body")
    return body
