# Lets try to solve the problem for minimum number of coins to make amount using  n coins
# the coins we have are [1,3,4]
from gettext import dpgettext


# def coin_change_td(n, coins, memo={}):
#     if n == 0:
#         return 0
#     if n < 0:
#         return float('inf')
#     if n in memo:
#         return memo[n]
#     memo[n] = min(coin_change_td(n-c, coins, memo) + 1 for c in coins)
#     return memo[n]
#
# print(coin_change_td(6, [1, 3, 4]))  # -> 2 (3+3)

#  ^^^^  didn't understand a thing.

# def coin_change_tab(n: int, coins: list[int]) -> int:
#     dp = [float('inf')] * (n+1)
#     dp[0] = 0
#     for i in range(1,n+1):
#         for c in coins:
#             if i-c >= 0:
#                 dp[i] = min(dp[i], dp[i-c] + 1)
#     return dp[n]
#
# if __name__ == "__main__":
#     print(coin_change_tab(6,[1,2,5]))
#


# Alternative Approach

