# -*- coding: utf-8 -*-


def spiral_order_print(arr: list):
    """
    """
    sr, sc = 0, 0
    er, ec = len(arr) - 1, len(arr[-1]) - 1
    while sr <= er and sc <= ec:
        print_edge(arr, sr, sc, er, ec)
        sr += 1
        sc += 1
        er -= 1
        ec -= 1


def print_edge(arr: list, srow, scol, erow, ecol):
    if srow == erow:
        for i in range(scol, ecol):
            print(arr[srow][i])
    elif scol == ecol:
        for i in range(srow, erow):
            print(arr[i][scol])
    else:
        cur_row, cur_col = srow, scol
        while cur_col != ecol:
            print(arr[srow][cur_col])
            cur_col += 1
        while cur_row != erow:
            print(arr[cur_row][ecol])
            cur_row += 1
        while cur_col != scol:
            print(arr[erow][cur_col])
            cur_col -= 1
        while cur_row != srow:
            print(arr[cur_row][scol])
            cur_row -= 1


if __name__ == '__main__':
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    spiral_order_print(a)
