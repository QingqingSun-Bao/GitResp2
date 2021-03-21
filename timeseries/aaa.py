
def func(num):
    if num == 1:
        return 1
    else:
        return num * func(num-1)
print(func(998))

if __name__ == "__main__":
    func