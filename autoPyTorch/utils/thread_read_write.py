
import fasteners, json, os, threading

thread_lock = threading.Lock()

def write(filename, content):
    with open(filename, 'w+') as f:
        f.write(content)

def read(filename):
    content = '{}'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
    return content

def append(filename, content):
    with fasteners.InterProcessLock('{0}.lock'.format(filename)):
        with open(filename, 'a+') as f:
            f.write(content)

def update_results_thread(filename, info):
    thread_lock.acquire()
    with fasteners.InterProcessLock('{0}.lock'.format(filename)):
        content = json.loads(read(filename))
        name = info['name']
        result = info['result']
        refit_config = info['refit_config']
        text = info['text']
        seed = str(info['seed'])

        infos = content[name] if name in content else dict()
        infos[seed] = {'result': result, 'description': text, 'refit': refit_config}
        content[name] = infos

        write(filename, json.dumps(content, indent=4, sort_keys=True))
    thread_lock.release()


def update_results(filename, info):
    thread = threading.Thread(target = update_results_thread, args = (filename, info))
    thread.start()