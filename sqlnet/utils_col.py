def degrad_c(c):
    ch = get_last_char(c)
    inx = int(ch) + 1
    return c[:len(c) - 1] + str(inx)

def get_last_char(c):
    return c[len(c) - 1]

if __name__ == '__main__':
    print(degrad_c('col_3'))