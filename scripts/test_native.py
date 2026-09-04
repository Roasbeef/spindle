#!/usr/bin/env python3
"""Real helper tests. Model configuration is mandatory, never a silent skip."""
import math
import os
import signal
import struct
import subprocess
import tempfile
import unittest

signal.alarm(120)
HELPER = os.environ['SPINDLE_HELPER']
MODEL = os.environ['SPINDLE_MODEL']

class Peer:
    def __init__(self):
        self.log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([HELPER, MODEL], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log)
    def read(self):
        header = self.proc.stdout.read(4)
        if len(header) != 4:
            raise AssertionError('helper exited before response')
        size, = struct.unpack('>I', header)
        if not 0 < size <= 1048576:
            raise AssertionError('unbounded response')
        body = self.proc.stdout.read(size)
        if len(body) != size:
            raise AssertionError('truncated response')
        return body
    def send(self, body):
        self.proc.stdin.write(struct.pack('>I', len(body)) + body)
        self.proc.stdin.flush()
    def request(self, tag, ident, texts):
        encoded = [text.encode() for text in texts]
        self.send(struct.pack('>BII', tag, ident, len(texts)) +
                  b''.join(struct.pack('>I', len(text)) + text for text in encoded))
    def stop(self):
        self.send(struct.pack('>BI', 3, 0))
        return self.proc.wait(timeout=5)
    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.log.close()

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.peer = Peer()
        self.addCleanup(self.peer.close)
        tag, version, dims, context = struct.unpack('>BHII', self.peer.read())
        self.assertEqual((tag, version, dims, context), (0, 1, 768, 2048))
    def test_batch_vectors_are_normalized_and_repeatable(self):
        texts = ['task: search result | query: durable agent memory',
                 'title: notes | text: Decisions are preserved in durable notes.']
        self.peer.request(1, 7, texts)
        first = self.peer.read()
        self.assertEqual(struct.unpack('>BIII', first[:13]), (129, 7, 2, 768))
        floats = struct.unpack('>1536f', first[13:])
        for i in range(2):
            row = floats[i*768:(i+1)*768]
            self.assertTrue(all(math.isfinite(x) for x in row))
            self.assertAlmostEqual(sum(x*x for x in row), 1, places=5)
        self.peer.request(1, 7, texts)
        self.assertEqual(self.peer.read(), first)
        self.assertEqual(self.peer.stop(), 0)
    def test_input_overflow_is_an_error_and_engine_remains_usable(self):
        self.peer.request(1, 8, ['token ' * 5000])
        response = self.peer.read()
        self.assertEqual(response[0], 255)
        self.assertIn(b'token limit', response)
        self.peer.request(2, 9, ['hello'])
        self.assertEqual(self.peer.read()[0], 130)
        self.assertEqual(self.peer.stop(), 0)
    def test_malformed_request_is_not_truncated_success(self):
        self.peer.send(struct.pack('>BII', 1, 10, 17))
        self.assertEqual(self.peer.read()[0], 255)
        self.assertEqual(self.peer.stop(), 0)
    def test_oversized_frame_exits_without_reading_its_body(self):
        self.peer.proc.stdin.write(struct.pack('>I', 1048577))
        self.peer.proc.stdin.flush()
        self.assertEqual(self.peer.proc.wait(timeout=5), 65)
    def test_eof_terminates_the_loaded_helper(self):
        self.peer.proc.stdin.close()
        self.assertEqual(self.peer.proc.wait(timeout=5), 0)
    def test_shutdown_interrupts_a_large_batch(self):
        self.peer.request(1, 11, ['token ' * 1500] * 16)
        self.assertEqual(self.peer.stop(), 0)

class StartupTests(unittest.TestCase):
    def test_shutdown_before_model_greeting(self):
        peer = Peer()
        self.addCleanup(peer.close)
        self.assertEqual(peer.stop(), 0)
    def test_eof_before_model_greeting(self):
        peer = Peer()
        self.addCleanup(peer.close)
        peer.proc.stdin.close()
        self.assertEqual(peer.proc.wait(timeout=5), 0)

if __name__ == '__main__':
    unittest.main()
