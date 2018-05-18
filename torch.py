
a = torch.randn(1,2,3,4,5)
print a
print torch.numel(a)

b = torch.zeros(4,4)
print b
print torch.numel(b)

c = torch.linspace(9, 99, 9)
print c

d = torch.logspace(-9, 9, steps = 9)
print d

e = torch.eye(4, 3)
print e

f = torch.ones(3, 6)
print f

tmp = torch.FloatTensor(2, 3)
g = torch.ones_like(tmp)
print g

h = torch.arange(1, 9, 0.9)
print h

i = torch.zeros(3, 4)
print i

j = torch.zeros_like(tmp)
print j

print tmp
k = torch.cat((tmp, tmp, tmp), 0)
print k
k = torch.cat((tmp, tmp, tmp), 1)
print k

l = torch.chunk(k, 2, 0)
print l
l = torch.chunk(k, 9, 1)
print l

tmp = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print tmp
m = torch.gather(tmp, 0, torch.LongTensor([[0, 0, 0], [1, 0, 0]]))
print m
m = torch.gather(tmp, 1, torch.LongTensor([[0, 0, 0], [1, 0, 0]]))
print m

tmp = torch.randn(3, 4)
print tmp
n = torch.LongTensor([0, 2])
print n
n = torch.index_select(tmp, 0, n)
print n
n = torch.LongTensor([0, 2])
n = torch.index_select(tmp, 1, n)
print n

print tmp
mask = tmp.ge(0.5)
print mask
o = torch.masked_select(tmp, mask)
print o

# torch.nonzero: return the indices of a non-zero element in input
p = torch.nonzero(torch.Tensor([1, 1, 1, 0, 1]))
print p
p = torch.nonzero(torch.Tensor([[0.6, 0.0, 0.0, 0.0],
                                [0.0, 0.4, 0.0, 0.0],
                                [0.0, 0.0, 1.2, 0.0],
                                [0.0, 0.0, 0.0, -0.4]]))
print p

q = torch.split(p, 1, 0)
print q
q = torch.split(p, 1, 1)
print q

tmp = torch.zeros(2,1,2,1,2)
print tmp
print tmp.size()
r = torch.squeeze(tmp)
print r
print r.size()
print tmp
print tmp.size()
r = torch.squeeze(tmp, 0)
print r
print r.size()
r = torch.squeeze(tmp, 1)
print r
print r.size()

tmp = torch.randn(3, 5)
print tmp
s = torch.t(tmp)
print s

t = torch.take(tmp, torch.LongTensor([0,1,7,9,12]))
print t

u = torch.clamp(t, 0, 1)
print u

tmp = torch.randn(3, 4)
print tmp
v = torch.mean(tmp, 0)
print v
v = torch.mean(tmp, 1)
print v

print tmp
w = torch.median(tmp)
print w
w = torch.median(tmp, 0)
print w
w = torch.median(tmp, 1)
print w

print tmp
x = torch.max(tmp)
print x
x = torch.max(tmp, 0)
print x
x = torch.max(tmp, 1)
print x

print tmp
y = torch.min(tmp)
print y
y = torch.min(tmp, 0)
print y
y = torch.min(tmp, 1)
print y

# sum, std, prod, var, sort, dot
