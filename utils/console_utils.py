

def input_int(s, default_value, valid_list=None):
    try:
        inp = input(s)        
        i = int(inp)
        if (valid_list is not None) and (i not in valid_list):
            return default_value
        return i
    except:
        return default_value
        
def input_bool(s, default_value):
    try:
        return bool ( {"y":True,"n":False,"1":True,"0":False}.get(input(s).lower(), default_value) )
    except:
        return default_value
        
def input_str(s, default_value, valid_list=None):
    try:
        inp = input(s)
        if (valid_list is not None) and (inp.lower() not in valid_list):
            return default_value
        return inp
    except:
        return default_value