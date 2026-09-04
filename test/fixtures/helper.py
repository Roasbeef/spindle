#!/usr/bin/env python3
"""A deterministic protocol peer for owner and cancellation tests."""
import struct
import sys
import time

mode = sys.argv[1]
def read_exact(n):
    data = sys.stdin.buffer.read(n)
    if len(data) != n:
        sys.exit(0)
    return data

def send(body):
    sys.stdout.buffer.write(struct.pack('>I', len(body)) + body)
    sys.stdout.buffer.flush()

if mode in ('delayed-greeting', 'startup-error'):
    time.sleep(0.05)

if mode not in ('no-greeting', 'startup-error-on-stop', 'startup-error'):
    send(struct.pack('>BHII', 0, 1, 2, 2048))
if mode == 'startup-error':
    reason = b'unsupported pooling'
    send(struct.pack('>BII', 255, 0, len(reason)) + reason)

while True:
    size, = struct.unpack('>I', read_exact(4))
    body = read_exact(size)
    command, ident = struct.unpack('>BI', body[:5])
    if command == 3:
        if mode == 'ignore-stop':
            continue
        if mode == 'garbage-on-stop':
            send(b'bad')
            sys.exit(0)
        if mode == 'startup-error-on-stop':
            send(struct.pack('>BII', 255, 0, 4) + b'load')
            sys.exit(74)
        sys.exit(74 if mode == 'stall-exit-error' else 0)
    if mode in ('stall', 'stall-exit-error', 'no-greeting', 'garbage-on-stop'):
        continue
    count, = struct.unpack('>I', body[5:9])
    if mode == 'wrong-id':
        ident += 1
    if command == 1:
        send(struct.pack('>BIII', 129, ident, count, 2) + struct.pack('>ff', 1, 0) * count)
    elif command == 2:
        send(struct.pack('>BII', 130, ident, count) + struct.pack('>I', 3) * count)
