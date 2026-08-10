import numba

def auto_jit_rule(user_func):
    try:
        compiled_func = numba.njit(user_func, fastmath=True)
        return compiled_func, True
    except Exception as e:
        print(f"JIT compilation warning: executing as standard Python. ({e})")
        return user_func, False
