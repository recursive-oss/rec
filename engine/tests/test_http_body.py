import io
import unittest
from email.message import Message

from rec_mlx.http_body import MAX_BODY, MAX_LINE, read_body


class BodyTest(unittest.TestCase):
    def read(self, body, *headers):
        message = Message()
        for name, value in headers:
            message[name] = value
        return read_body(io.BytesIO(body), message)

    def test_fixed_length(self):
        self.assertEqual(self.read(b"abc", ("Content-Length", "3")), b"abc")
        self.assertEqual(self.read(b"abc", ("Content-Length", " 3\t")), b"abc")
        self.assertEqual(
            self.read(b"x" * MAX_BODY, ("Content-Length", str(MAX_BODY))), b"x" * MAX_BODY
        )

    def test_chunk_extensions_and_trailers(self):
        self.assertEqual(
            self.read(
                b"1;foo=bar\r\na\r\n2\r\nbc\r\n0\r\nContent-Length: 99\r\n\r\n",
                ("Transfer-Encoding", "CHUNKED"),
            ),
            b"abc",
        )

    def test_rejects_bad_lengths_and_ambiguous_framing(self):
        cases = [
            (),
            (("Content-Length", "-1"),),
            (("Content-Length", "+1"),),
            (("Content-Length", "0"),),
            (("Content-Length", "2"),),  # truncated
            (("Content-Length", str(MAX_BODY + 1)),),
            (("Content-Length", "1"), ("Content-Length", "1")),
            (("Content-Length", "1"), ("Transfer-Encoding", "chunked")),
            (("Transfer-Encoding", "gzip, chunked"),),
            (("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "chunked")),
        ]
        for headers in cases:
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                self.read(b"x", *headers)

    def test_rejects_malformed_or_unbounded_chunks(self):
        for body in (
            b"0\r\n\r\n",
            b"-1\r\n",
            b"+1\r\n",
            b" 1\r\n",
            b"g\r\n",
            b"1\na\n",
            b"1\r\n",
            b"1\r\naXX",
            b"1\r\na\r\n0\r\n",
            b"0\r\nbad\r\n\r\n",
            b"800001\r\n",
            b"1;" + b"a" * MAX_LINE + b"\r\n",
            b"1\r\na\r\n" * 14000 + b"0\r\n\r\n",
        ):
            with self.subTest(body=body[:40]), self.assertRaises(ValueError):
                self.read(body, ("Transfer-Encoding", "chunked"))
