import datetime


def make_timestamp(style: str = 'YMD'):
    styles = {
        'DMY': "%d-%m-%Y-%H%M%S",
        'MDY': "%m-%d-%Y-%H%M%S",
        'YMD': "%Y-%m-%d-%H%M%S"
    }
    return datetime.datetime.now().strftime(styles[style])
