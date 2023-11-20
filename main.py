import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = 8
batch_size = 4

with open(r'C:\Users\cembi\PycharmProjects\heLaLceM\dataset\wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:200])
chars = sorted(set(text))
print(chars)
print(len(chars))
vocabulary_size = len(chars)

string_to_int = {ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

print(encode('hello'))

encoded_hello = encode('hello')
decoded_hello = decode(encoded_hello)
print(decoded_hello)

data = torch.tensor(encode(text), dtype = torch.long)
print(data[:100])

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('When input is', context, 'target is', target)
