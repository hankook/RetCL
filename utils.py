import os, logging

def set_logging_options(logdir):
    handlers = [logging.StreamHandler(os.sys.stdout)]
    if logdir is not None:
        handlers.append(logging.FileHandler(os.path.join(logdir, 'log.txt')))
    logging.basicConfig(format="[%(asctime)s - %(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=handlers)

