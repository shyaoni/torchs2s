
class a():
    def produce(self):
        return type(self)(4)
    
class b(a):
    def __init__(self, x):
        self.x = x

    def f(self):
        return super().produce()

c = b(1)
d = c.f()

print(type(d), d.x)
