import threading
from datetime import datetime
from time import sleep
from tqdm import tqdm


def run_in_thread(fn):
    def run(*k, **kw):
        t = threading.Thread(target=fn, args=k, kwargs=kw)
        t.start()

    return run


def pbar(res_handles, n_avg, n_label, timeout=10, return_times=False):
    n_now = 0
    n = None
    m = 0
    while n is None:
        if m * 0.1 > timeout:
            print("reached timeout")
            break
        sleep(0.1)
        n = res_handles.get(n_label).fetch_all()

    times_vec = []
    with tqdm(total=n_avg, desc=n_label) as pbar_obj:
        while n < n_avg:
            n = res_handles.get(n_label).fetch_all() + 1
            sleep(0.1)
            if n is not None and n > n_now:
                pbar_obj.update(n - n_now)
                n_now = n
                if return_times:
                    times_vec.append(datetime.now())
    if return_times:
        return times_vec
