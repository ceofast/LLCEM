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
torch.exp(out)
# Bu kısımda daha gelişmiş tensor işlemleri gerçekleştiriliyor: multinominal dağılımdan örnekler çekme,
# tensorleri birleştirme, alt üçgen ve üst üçgen matrisler oluşturma ve maskeleme işlemleri yapılıyor.
###############################################################################################################
input = torch.zeros(2, 3, 4)
out = input.transpose(0, 2)
out.shape

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

# Stack the tensors along a new dimension
stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
stacked_tensor
##############################################################################################################
import torch.nn as nn

sample = torch.tensor([10, 10, 10.])
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))

import torch.nn.functional as F

# Create a tensor
tensor1 = torch.tensor([1.0, 2.0, 3.0])
# Apply softmax using torch.nn.functional.softmax()
softmax_output = F.softmax(tensor1, dim=0)
print(softmax_output)
############################################################################################################
# nn.Embedding PyTorch'ta, doğal dil işleme (NLP) ve diğer makine öğrenimi uygulamalarında sıkça kullanılan
# bir yapıdır. Bu terim, genellikle derin öğrenme kütüphanesi PyTorch'un bir parçası olan torch.nn modülü
# içinde bulunan bir sınıfı ifade eder. nn.Embedding, kelimeleri veya diğer türden öğeleri düşük boyutlu
# vektörler olarak temsil etmek için kullanılır. Bu, özellikle kelime işleme gibi görevlerde önemlidir
# çünkü modellerin metni anlamasını ve işlemesini sağlar.
#
# nn.Embedding sınıfının temel özellikleri şunlardır:
#
# Kelime Vektörleri: Her bir benzersiz kelimeyi (veya token'ı), önceden belirlenmiş boyutta bir vektörle
# eşler. Bu vektörler genellikle rastgele başlatılır ve eğitim sırasında güncellenir.
#
# Boyutlandırma: nn.Embedding, iki ana parametre alır: num_embeddings ve embedding_dim. num_embeddings,
# kelime haznesinin (veya token'ların) toplam sayısını belirtirken, embedding_dim, her bir kelime vektörünün
# boyutunu ifade eder.
#
# Eğitilebilirlik: Embedding katmanındaki vektörler, eğitim sürecinde geri yayılım (backpropagation)
# kullanılarak güncellenebilir. Bu sayede model, görevine özel anlamlı vektör temsilleri öğrenebilir.
#
# Verimlilik: Kelime vektörlerinin aranması ve işlenmesi oldukça verimlidir. nn.Embedding, verimli bir
# şekilde büyük kelime hazneleriyle çalışabilir.
#
# nn.Embedding kullanımı genellikle şu adımları içerir:
#
# Embedding katmanını başlatma (nn.Embedding(num_embeddings, embedding_dim) kullanarak).
# Kelime indekslerini (genellikle bir cümle veya dökümandan) embedding katmanına besleme.
# Geri dönen vektörleri modelin geri kalanında kullanma.
# Bu yapı, dil modelleri, metin sınıflandırma, duygu analizi ve daha birçok NLP görevinde yaygın olarak
# kullanılır.

# PyTorch'ta nn.Embedding sınıfı, kelime temsillerini yönetmek için kullanılır. Bu, her bir kelimeyi veya
# token'ı, öğrenilebilir bir vektörle eşler. Bu vektörler, genellikle kelime anlamlarını ve ilişkilerini
# kodlar ve dil işleme görevlerinde kullanılır.
#
# Örneğin, bir cümledeki her kelime, belirli bir boyutta bir vektör ile temsil edilir. Bu vektörler,
# modelin öğrenme süreci sırasında güncellenerek, kelimelerin anlamlarını ve birbirleriyle olan ilişkilerini
# daha iyi yansıtacak şekilde ayarlanır.
#
# Temel Parametreler
# num_embeddings: Kelime haznesindeki benzersiz kelime sayısı. Örneğin, 10.000 benzersiz kelimeniz varsa,
# bu değer 10.000 olur.
# embedding_dim: Her bir kelime vektörünün boyutu. Örneğin, her kelimeyi 300 boyutlu bir vektör ile temsil
# etmek istiyorsanız, bu değer 300 olur.

####################################
# 1. Embedding Katmanını Başlatma: #
####################################
import torch.nn as nn
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)

######################################################
# 2. Kelime İndekslerini Embedding Katmanına Besleme #
######################################################
# Diyelim ki, 'kedi' kelimesinin indeksi 23, 'koşuyor' kelimesinin indeksi 857. Bu kelimeler için vektörleri almak
# isterseniz:
indices = torch.tensor([23, 857])
embedded = embedding(indices)

######################################
# 3. Elde Edilen Vektörleri Kullanma #
######################################
# embedded değişkeni artık bu kelimelerin vektör temsillerini içerir. Bu vektörler, modelin diğer katmanlarına
# beslenebilir.

# Diyelim ki, kelime haznenizde 'elma', 'armut', 'muz' gibi meyveler var ve her birini 2 boyutlu bir vektörle
# temsil etmek istiyorsunuz. Bu durumda, nn.Embedding ile bu kelimeleri nasıl vektörlere dönüştürebileceğinizi
# görselleştirelim.

###################################
# 1. Kelime Haznesi ve İndeksleri #
###################################
# * Elma -> 0
# * Armut -> 1
# * Muz -> 2

#################################
# 2. Embedding Katmanı Başlatma #
#################################
embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)

#############################################
# 3. Kelime İndekslerini Vektörlere Çevirme #
#############################################

# * Elma (0) -> [1.5, -2.3]
# * Armut (1) -> [0.9, 3.1]
# * Muz (2) -> [-1.0, 0.5]

# Bu örneklerde, her kelime için rastgele başlatılmış 2 boyutlu vektörler varsayılmıştır.
# Gerçek uygulamalarda, bu vektörler eğitim sürecinde güncellenir ve daha anlamlı hale gelir.

# Kelime düzeyinde embedding kullanımının aksine, karakter düzeyinde embedding kullanımı, her bir karakteri
# düşük boyutlu bir vektörle temsil etmeyi içerir. Bu yaklaşım, özellikle karakter bazlı dil işleme görevleri
# için kullanışlıdır ve bazı avantajlar sunar:

# * 1. Daha Az Kelime Haznesi #
#  Kelime düzeyinde embeddinglerde, kelime haznesi genellikle binlerce hatta milyonlarca kelimeyi içerir.
#  Karakter düzeyinde ise, sadece alfabedeki karakter sayısı kadar bir kelime haznesine ihtiyaç vardır
#  (genellikle çok daha az).

# * 2. Yazım Hataları ve Yeni Kelimeler #
#  Karakter düzeyinde embeddingler, yazım hataları veya metinde ilk kez karşılaşılan kelimelerle daha iyi başa
#  çıkabilir, çünkü her kelimeyi karakterlerine ayırarak işler.

# * 3. Dil Özgünlüğü #
#  Bazı dillerde, kelime düzeyinde embedding kullanmak zor veya verimsiz olabilir. Özellikle Çince gibi karakter
#  tabanlı dillerde, karakter düzeyinde embeddingler daha pratik olabilir.


##########################################
# KARAKTER DÜZEYİNDE EMBEDDING KULLANIMI #
##########################################

# Karakter düzeyinde embedding kullanımında, her karakterin bir indeksi vardır ve bu indeksler, nn.Embedding katmanına
# beslenerek karakter vektörleri elde edilir.

# Diyelim ki, İngilizce alfabede 26 harf var ve her birini 2 boyutlu bir vektörle temsil etmek istiyorsunuz:

# * 1. Karakter İndeksleri
# * a -> 0
# * b -> 1
# * c -> 2
# ....
# * -> 25

# * 2. Embedding Katmanını Başlatma
embedding = nn.Embedding(num_embeddings=26, embedding_dim=2)

# * 3. Karakter İndekslerini Vektörlere Çevirme
# * a(0) -> [1.2, -0.9]
# * b(1) -> [0.5, 2.3]
# ...
# <(25) -> [-1.1, 0.4]

# Bu örneklerde, her karakter için rastgele başlatılmış 2 boyutlu vektörler varsayılmıştır. Gerçek uygulamalarda,
# bu vektörler eğitim süreçlerinde güncellenir.

import torch
import torch.nn as nn

# Inıtıalize an embedding layer
vocab_size = 10000
embedding_dim = 100
embedding = nn.Embedding(vocab_size, embedding_dim)

# Create some input indices
input_indices = torch.LongTensor([1, 5, 3, 2])

# Apply the embedding layer
embedded_output = embedding(input_indices)

print(embedded_output.shape)
print(embedded_output)
########################################################################################################################

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(a @ b)

########################################################################################################################
int_64 = torch.randint(1, (3, 2)).float()
# type int64
float_32 = torch.rand(2, 3)
# type float32
print(int_64.dtype, float_32.dtype)
result = torch.matmul(int_64, float_32)
print(result)