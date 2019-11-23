# -*-coding: utf-8 -*-

"""
"""

import time
import logging

from inspect import signature
from functools import wraps, partial


def timethis(func):
    @wraps(func)   # 保留原始函数的元数据，比如名字，文档字符串，注解和参数签名等，可以通过__doc__查看
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


def logged(func=None, *, level=logging.DEBUG, name=None, message=None):
    # 带可选参数的装饰器
    if func is None:
        return partial(logged, level=level, name=name, message=message)

    logname = name if name else func.__module__
    log = logging.getLogger(logname)
    logmsg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        log.log(level, logmsg)
        return func(*args, **kwargs)
    return wrapper


@logged
def add(x, y):
    return x + y


@logged(level=logging.CRITICAL, name="example")
def spam():
    print("Spam function")

# ------------给装饰器增加属性------


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def typeassert(*ty_args, **ty_kwargs):
    """ 类型检查装饰器
    """
    def decorate(func):
        # If in optimized mode, disable type checking
        if not __debug__:
            return func

        # 提取一个可调用对象的参数签名信息
        sig = signature(func)
        # bind_partial() 方法来执行从指定类型到名称的部分绑定
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            # bind 和 bind_partial很类似，不允许忽略任何参数
            bound_values = sig.bind(*args, **kwargs).arguments
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            f'Argument {name} must be {bound_types[name]}')
            return func(*args, **kwargs)
        return wrapper
    return decorate


if __name__ == '__main__':
    @typeassert(int, z=int)
    def spam(x, y, z=42):
        print(x, y, z)

    spam(1, 2, 3.0)
    from functools import lru_cache
    a = 5