def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

n = int(input("Enter the value of n: "))
prime_numbers = []
num = 2
while len(prime_numbers) < n:
    if is_prime(num):
        prime_numbers.append(num)
    num += 1

print(f"The first {n} prime numbers are: {prime_numbers}")