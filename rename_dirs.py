import os
import re
import multiprocessing as mp
import time


def printinteger(a: int):
    time.sleep(5)
    print(f"My number is {a}")
    return a

if __name__ == "__main__":
    start_time = time.time()
    integers = list(range(4))
    print(f"NUmber of cpus: {mp.cpu_count()}")
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(printinteger, args=(a,)) for a in integers]
    results = [r.get() for r in results]
    pool.close()
    pool.join()
    result = sum(results)
    print(f"Sum is equal: {result}")
    end_time = time.time()
    print(f"Results: {results}, Time: {end_time - start_time:.2f} seconds")




