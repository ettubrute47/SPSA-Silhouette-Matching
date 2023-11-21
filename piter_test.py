# %%
import multiprocessing as mp
import time
from contextlib import contextmanager
import numpy as np


# %%
def timely_iter(sleep_time):
    for i in range(3):
        time.sleep(sleep_time)
        yield i


def pzip1(*iterators, num_workers=4):
    # will execute each iterator in parallel
    # Pool.imap
    # so either pool.map per iterator
    # or zip a pool.imap on iterators
    # test if stopiteration works...
    with mp.Pool(num_workers) as pool:
        while True:
            yield pool.map(next, iterators)


def pzip2(*iterators, num_workers=4):
    # will execute each iterator in parallel
    # Pool.imap
    # so either pool.map per iterator
    # or zip a pool.imap on iterators
    # test if stopiteration works...
    with mp.Pool(num_workers) as pool:
        piters = [pool.imap(next, iterator) for iterator in iterators]
        while True:
            vals = list(map(next, piters))
            yield vals


# %%
# iterators = [timely_iter(np.random.rand() * 0.5 + 0.2) for i in range(5)]
# for i, *vals in zip(range(2), *iterators):
#     print(i, vals)

# %%
# this hangs
# iterator_args = [np.random.rand()*0.2 + 0.1 for i in range(5)]
# with mp.Pool(4) as pool:
#     piter = pool.map(timely_iter, iterator_args)
#     print(piter)
#     for val in piter:
#         print(val)

# %% [markdown]
# So the idea is that function puts into queue, so something is started... then it yields from it

# %%
# from concurrent.futures import ProcessPoolExecutor

# def run_generator(t):
#     return timely_iter(t)

# with ProcessPoolExecutor() as executor:
#     result = executor.submit(run_generator, 0.1)
#     print(result.result())


# %%
def execute_iterator(q):
    # for i in range(3):
    #     q.put(i)
    iterator = timely_iter(1)
    while True:
        end_signal = q.recv()
        if end_signal:
            break
        try:
            val = next(iterator)
        except StopIteration:
            break
        q.send(val)

    # for el in timely_iter(1):
    #     q.send(el)
    #     end_signal = q.recv()
    #     if end_signal:
    #         break


if __name__ == "__main__":
    from mypar import pzip

    start = time.perf_counter()
    num_workers = 3
    connectors = [mp.Pipe() for i in range(num_workers)]
    # well I can start it
    procs = [
        mp.Process(target=execute_iterator, args=(pipe[1],)) for pipe in connectors
    ]
    for p, (pipe, _) in zip(procs, connectors):
        p.start()
    for i in range(2):
        # prime
        for j, (pipe, _) in enumerate(connectors):
            pipe.send(False)
        for j, (pipe, _) in enumerate(connectors):
            print(i, j, pipe.recv())
    for j, (pipe, _) in enumerate(connectors):
        pipe.send(True)  # kill
    print("Duration: ", time.perf_counter() - start)
    for p, (pipe, _) in zip(procs, connectors):
        p.join()
        # while pipe.poll():
        #     print("AFTER? ", i, j, pipe.recv())
    print("Duration: ", time.perf_counter() - start)

    from functools import partial

    starters = [partial(timely_iter, 1) for i in range(3)]
    start_time = time.perf_counter()
    with pzip(*starters, buffer=2) as iterator:
        for i, vals in zip(range(2), iterator):
            print(vals)
        print("Done pzip?")
    print("Duration: ", time.perf_counter() - start_time)
    print("Done everything?")

# %%
