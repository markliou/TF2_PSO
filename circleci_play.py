print("hello, circleci")

class test():
    def __init__(self, i):
        self.i = i 
        self.j = self.add2(i)
    pass 
    def add2(self, i):
        return i+2
    pass

def main():
    t = test(2)
    print(t.j)
pass 

if __name__ == "__main__":
    main()
pass