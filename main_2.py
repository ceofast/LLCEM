import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

randint = torch.randint(-100, 100, (6,))
randint
# Burada, 6 elemanlı, -100 ile 100 arasında rastgele tam sayılar içeren bir tensor oluşturulur.
############################################################################################################
tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor
zeros = torch.zeros(2, 3)
zeros
ones = torch.ones(3, 4)
ones
input = torch.empty(2, 3)
input
arange = torch.arange(5)
arange
linspace = torch.linspace(3, 10, steps = 5)
linspace
logspace = torch.logspace(start=-10, end=10, steps=5)
logspace
eye = torch.eye(5)
eye
a = torch.empty((2,3), dtype=torch.int64)
empty_like = torch.empty_like(a)
empty_like
# Bu kısımda çeşitli tensorler oluşturuluyor: Belirli değerlere, sıfırlarla, birlerle, boş, aralıklı
# değerlerle, logaritmik aralıklı değerlerle dolu tensorler ve birim matris.
###############################################################################################################
# Zaman Ölçümü ve Büyük Tensör İşlemleri
import torch
import numpy as np
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

%%time
start_time = time.time()
# matrix operation here
zeros = torch.zeros(1, 1)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")
# Using thr time library to record execution time %%time to record the time taken for the cell to execute
# Burada, bir tensor oluşturma işleminin ne kadar sürede tanımlandığını ölçmek için 'time' kütüphanesi
# kullanılıyor.
###############################################################################################################
torch_rand1 = torch.rand(100, 100, 100, 100).to(device)
torch_rand2 = torch.rand(100, 100, 100, 100).to(device)
np_rand1 = torch.rand(100, 100, 100, 100)
np_rand2 = torch.rand(100, 100, 100, 100)

start_time = time.time()

rand = (torch_rand1 @ torch_rand2)
print(f"{elapsed_time:.8f}")


start_time = time.time()

rand = np.multiply(np_rand1, np_rand2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")
# Bu bölümde, büyük boyutlu tensorler üzerinde işlem yapılıyor ve bu işlemlerin süreleri ölçülüyor.
# 'torch_rand1 @ torch_rand2' ifadesi, iki büyük tensorun matris çarpımını gerçekleştirir.
#############################################################################################################
# torch.stack, torch.multinominal, torch.tril, torch.triu, input.T / input.transpose, nn.Linear, torch,cat,
# F.softmax (show all the

# Define a probability tensor
probabilities = torch.tensor([0.1, 0.9])
# 10% or 0.1 => 0, %90 => 1. each probability points to the index of the probability in the tensor
# Draw 5 samples from multinominal distribution
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)
tensor = torch.tensor([1, 2, 3, 4])
out = torch.cat((tensor, torch.tensor([5])), dim = 0)
out
out = torch.tril(torch.ones(5, 5))
out
out = torch.triu(torch.ones(5, 5))
out
out = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0, float('-inf'))
out
# Bu kısımda daha gelişmiş tensor işlemleri gerçekleştiriliyor: multinominal dağılımdan örnekler çekme,
# tensorleri birleştirme, alt üçgen ve üst üçgen matrisler oluşturma ve maskeleme işlemleri yapılıyor.


