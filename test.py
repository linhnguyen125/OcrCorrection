import lmdb


# Tạo môi trường LMDB
env = lmdb.open('mydatabase')  # Tạo cơ sở dữ liệu với kích thước tối đa 100 MB

# Bắt đầu một transaction ghi
with env.begin(write=True) as txn:
    # Ghi một cặp key-value vào cơ sở dữ liệu
    txn.put(b'username', b'linh2')
    txn.put(b'password', b'secret1232')

    # Bạn cũng có thể ghi nhiều cặp key-value khác nhau
    txn.put(b'email', b'linh@example.com2')

# Bắt đầu một transaction đọc
with env.begin() as txn:
    # Đọc dữ liệu từ cơ sở dữ liệu bằng khóa (key)
    username = txn.get(b'username')
    password = txn.get(b'password')
    email = txn.get(b'email')

    # In ra các giá trị đã đọc
    print(f"Username: {username.decode('utf-8')}")
    print(f"Password: {password.decode('utf-8')}")