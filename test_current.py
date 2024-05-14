import concurrent.futures

def my_function(x):
    return x * x

# ThreadPoolExecutor の場合
with concurrent.futures.ThreadPoolExecutor() as executor:
    # 関数に渡す引数のリスト
    args = [1, 2, 3, 4, 5]
    
    # map() を使って関数を並行実行し、結果を受け取る
    results = executor.map(my_function, args)
    
    for result in results:
        print(result)