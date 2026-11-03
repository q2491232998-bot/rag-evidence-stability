import argparse
import json
import os
import subprocess
import sys
import time

from filelock import FileLock

from common import ROOT


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timeout', type=float, default=86400)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    assert args.command and args.timeout > 0 and args.poll_seconds > 0
    deadline = time.monotonic() + args.timeout
    with FileLock(str(ROOT / 'logs/gpu_experiment.lock'), timeout=args.timeout):
        previous = None
        last_log = 0.
        while True:
            if (ROOT / 'logs/research.pause').exists():
                print(json.dumps({'event': 'paused_before_launch', 'pid': os.getpid()}), flush=True)
                return 75
            active = gpu_processes()
            now = time.monotonic()
            if active != previous or now - last_log >= 300:
                print(json.dumps({'event': 'waiting_for_idle_gpu' if active else 'gpu_idle', 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid(), 'gpu_pids': active}), flush=True)
                previous, last_log = active, now
            if not active:
                time.sleep(2)
                if not gpu_processes():
                    break
            if now >= deadline:
                raise TimeoutError('GPU remained occupied; no compute command launched')
            time.sleep(min(args.poll_seconds, max(.01, deadline - time.monotonic())))
        if (ROOT / 'logs/research.pause').exists():
            return 75
        print(json.dumps({'event': 'start', 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'command': args.command, 'isolation': 'No GPU compute processes at launch; external jobs are not controlled'}), flush=True)
        result = subprocess.run(args.command, cwd=ROOT)
        print(json.dumps({'event': 'exit', 'code': result.returncode, 'gpu_pids_after': gpu_processes()}), flush=True)
        return result.returncode


if __name__ == '__main__':
    sys.exit(main())
