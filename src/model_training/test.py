print(__file__)
print(f"{__file__[:len(__file__)-22]}")

try:
    print(1/0)
except Exception as err:
    print(err)