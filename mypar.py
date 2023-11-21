import multiprocessing as mp
from contextlib import contextmanager


def wrap_iterator(iter_start, q, num_steps=1):
    iterator = iter_start()
    check_signal = 0
    while True:
        check_signal += 1
        if check_signal % num_steps == 0:
            end_signal = q.recv()
            if end_signal:
                break
            check_signal = 0
        try:
            val = next(iterator)
        except StopIteration:
            print("Stop iteration")
            q.send(None)
            break

        q.send(val)
    print("Closing my pipe")
    q.close()


def iter_pipes(connectors, num_steps=1):
    # buffer N steps ahead
    send_signal = 0
    while True:
        send_signal += 1
        # prime them
        if send_signal % num_steps == 0:
            for i, (pipe, _) in enumerate(connectors):
                pipe.send(False)
            send_signal = 0
        try:
            yield tuple(pipe.recv() for (pipe, _) in connectors)
        except EOFError:
            print("EOFERROR ending?")
            break


@contextmanager
def pzip(*iterator_starters, buffer=1):
    if len(iterator_starters) > 8:
        raise ValueError("Too many iterators tried to start in parallel")
    connectors = [mp.Pipe() for i in range(len(iterator_starters))]
    # well I can start it
    procs = [
        mp.Process(
            target=wrap_iterator,
            args=(gen_iter, pipe[1], buffer),
        )
        for gen_iter, pipe in zip(iterator_starters, connectors)
    ]
    for i, (p, (pipe, _)) in enumerate(zip(procs, connectors)):
        try:
            p.start()
        except Exception as e:
            print("In exception oh no")
            for j in range(i):
                connectors[j][0].send(True)  # force to shutdown
                procs[j].terminate()
            raise e
    try:
        yield iter_pipes(connectors, buffer)
    finally:
        for j, (pipe, _) in enumerate(connectors):
            print("Sending end signal")
            pipe.send(True)  # kill
