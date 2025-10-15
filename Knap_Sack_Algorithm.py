# def knapsack_memoization(capacity, n):
#     print(f"knapsack_memoization({n}, {capacity})")
#     if memo[n][capacity] is not None:
#         print(f"Using memo for ({n}, {capacity})")
#         return memo[n][capacity]
#
#     if n == 0 or capacity == 0:
#         result = 0
#     elif weights[n - 1] > capacity:
#         result = knapsack_memoization(capacity, n - 1)
#     else:
#         include_item = values[n - 1] + knapsack_memoization(capacity - weights[n - 1], n - 1)
#         exclude_item = knapsack_memoization(capacity, n - 1)
#         result = max(include_item, exclude_item)
#
#     memo[n][capacity] = result
#     return result
#
#
# values = [300, 200, 400, 500]
# weights = [2, 1, 5, 3]
# capacity = 10
# n = len(values)
#
# memo = [[None] * (capacity + 1) for _ in range(n + 1)]
#
# print("\nMaximum value in Knapsack =", knapsack_memoization(capacity, n))

# a_string = "sdjbfisgffifuiabuifabfbsubdfousdufhsodfbsjdfbef"
# unique_alphabets = set()
# word_counter = dict()
# for char in a_string:
#     if char not in unique_alphabets:
#         unique_alphabets.add(char)
#         word_counter[char] = 1
#     else:
#         word_counter[char] += 1
#
# rev_wc = {a:b for b,a in word_counter.items()}
#
# print(word_counter)
# print(rev_wc)
#
# a = max(key for key, val in rev_wc.items() if key % 2 == 0)
# b = max(key for key, val in rev_wc.items() if key % 2 != 0)
#
# print(abs(a-b))

# class Solution:
#
#     def maxDifference(self, s: str) -> int:
#         unique_characters = set()
#         characters_dict = dict()
#         for char in s:
#             if char not in unique_characters:
#                 unique_characters.add(char)
#                 characters_dict[char] = 1
#             else:
#                 characters_dict[char] += 1
#
#         rev_dic = {a: b for b, a in characters_dict.items()}
#
#         highest_even = max(key for key, value in rev_dic.items() if key % 2 == 0)
#         highest_odd = max(key for key, value in rev_dic.items() if key % 2 != 0)
#
#         return abs(highest_even - highest_odd)
#
#
#
#
# if __name__ == "__main__":
#     sol = Solution()
#     S = "aaaaaaaabbbbb"
#     print(sol.maxDifference(S))


from collections import Counter


class Solution:

    def maxDifference(self, s: str) -> int:
        freq = Counter(s)
        counts = list(freq.values())

        max_even = max((c for c in counts if c % 2 == 0), default=0)
        max_odd = max((c for c in counts if c % 2 != 0), default=0)

        if not max_even or max_odd:
            return 0

        return
        abs(max_even - max_odd)


if __name__ == "__main__":
    sol = Solution()
    S = "aadadaibsbbbsibisbdbdbdqqqqdbbibcccc"
    sol.maxDifference(S)
