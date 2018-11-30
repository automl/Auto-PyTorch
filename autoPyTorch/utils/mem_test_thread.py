

import linecache
import os
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from threading import Thread
from time import sleep
from resource import getrusage, RUSAGE_SELF


class MemoryLogger():
    def __init__(self):
        self.queue = None
        self.monitor_thread = None

    def start(self, poll_interval=0.5):
        self.queue = Queue()
        self.monitor_thread = Thread(target=memory_monitor, args=(self.queue, poll_interval))
        self.monitor_thread.start()

    def stop(self):
        self.queue.put('stop')
        self.monitor_thread.join()


def memory_monitor(command_queue: Queue, poll_interval=1):
    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now())
                display_top(snapshot)

            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss
            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot, limit=1)
            print(datetime.now(), 'max RSS', old_max)

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-4:])
        print("#%s: %s:%s: %.1f KiB"
            % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))