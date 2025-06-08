n = int(input("Enter the number of rows for the upper triangle: "))
for i in range(1, n + 1):
    spaces = "  " * (n - i)
    stars = "* " * i
    print(spaces + stars)
