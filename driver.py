#!/usr/bin/python

import inspect
import time
import yoshi_euler

def timeProblem(f):
    start_time = time.time()
    result = f()
    return time.time() - start_time, result

def runProblemSuite(p=yoshi_euler):
    problems = {k: v for k, v in inspect.getmembers(p)
        if k.startswith('problem')}

    for k in sorted(problems):
        elapsed, a = timeProblem(problems[k])
        s = k.lstrip('problem')    # n = int(s[:4])
        print('{}: {:12} in {:.3f}s'.format(s, a, elapsed))

if __name__ == '__main__':
    runProblemSuite()
