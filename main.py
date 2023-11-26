# Bu kısım, PyTorch kütüphanesini içe katarıp eğer mevcutsa GPU'yu değilse CPU'yu işlem yapmak için kullanır.
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#######################################################################################################
block_size = 8
batch_size = 4

with open(r'C:\Users\cembi\PycharmProjects\heLaLceM\dataset\wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Burada ilk 200 karakteri ekrana yazdırdık.
print(text[:200])
#######################################################################################################
chars = sorted(set(text))
print(chars)
print(len(chars))
vocabulary_size = len(chars)
string_to_int = {ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
# Burada metindeki benzersiz karakterlerin bir seti oluşturulur ve bunlar sayısal değerlerle kodlanır.
# Bu, metin tabanlı verileri makine öğrenmesi modelleri için işlenebilir hale getirmek için yapılır.
#######################################################################################################
# Burada hello kelimesi kodlanır ve decodlanır. Bu kodlama ve dekodlama işlevlerin doğruluğunu
# test etmek içindir.
print(encode('hello'))
encoded_hello = encode('hello')
decoded_hello = decode(encoded_hello)
print(decoded_hello)
#######################################################################################################
# Tüm metin, sayısal değerlerle kodlanır ve bir PyTorch tensör'üne dönüştürülür. Daha sonra veri,
# eğitim ve doğrulama setleri olarak bölünür.
data = torch.tensor(encode(text), dtype = torch.long)
print(data[:100])

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]
#######################################################################################################
# Bu kısım, eğitim verileri üzerinde bir döngü oluşturur ve her adımda, belirli bir metin paröasının
# (context) sonraki karakterini (target) tahmin etmek için kullanılacağını gösterir.
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('When input is', context, 'target is', target)
# Genel olarak, bu kod, metin tabanlı verileri işlemek ve makine öğrenimi modeli eğitmek için hazırlamak
# amacıyla yazılmıştır. Özellikle, karakter düzeyinde dil modelleme veya benzeri bir görev için kullanılacaktır.
# When input is tensor([91]) target is tensor(48)
# When input is tensor([91, 48]) target is tensor(65)
# When input is tensor([91, 48, 65]) target is tensor(62)
# When input is tensor([91, 48, 65, 62]) target is tensor(1)
# When input is tensor([91, 48, 65, 62,  1]) target is tensor(44)
# When input is tensor([91, 48, 65, 62,  1, 44]) target is tensor(75)
# When input is tensor([91, 48, 65, 62,  1, 44, 75]) target is tensor(72)
# When input is tensor([91, 48, 65, 62,  1, 44, 75, 72]) target is tensor(67)

