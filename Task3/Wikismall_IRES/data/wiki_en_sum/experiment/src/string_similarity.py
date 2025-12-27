import textdistance


class StringSimilarity:

    def find_largest_common_substring(self, s1, s2):
        """
        Find the largest common substring between s1 and s2.
        :param s1: First string.
        :param s2: Second string.
        :return: Largest common substring and its length.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        longest, x_longest = 0, 0
        for x in range(1, m + 1):
            for y in range(1, n + 1):
                if s1[x - 1] == s2[y - 1]:
                    dp[x][y] = dp[x - 1][y - 1] + 1
                    if dp[x][y] > longest:
                        longest = dp[x][y]
                        x_longest = x
                else:
                    dp[x][y] = 0
        return s1[x_longest - longest: x_longest], longest

    def find_all_largest_common_substrings(self, s1, s2):
        """
        Find all largest common substrings between s1 and s2 by iteratively removing the found substrings.
        :param s1: First string.
        :param s2: Second string.
        :return: List of all largest common substrings.
        """
        common_substrings = []
        while True:
            lcs, lcs_len = self.find_largest_common_substring(s1, s2)
            if lcs_len <= 1:
                break
            common_substrings.append(lcs)
            s1 = s1.replace(lcs, '', 1)
            s2 = s2.replace(lcs, '', 1)
        return common_substrings

    def comm(self, s1, s2, common_substrs):
        """
        Compute the commonality between two strings based on the largest common substrings.
        :param s1: First string.
        :param s2: Second string.
        :param common_substrs: List of common substrings.
        :return: Commonality score.
        """
        sum_common_lengths = sum(len(substring)
                                 for substring in common_substrs)
        return 2 * sum_common_lengths / (len(s1) + len(s2))

    def unmatched_length(self, s, common_substrs):
        """
        Compute the total unmatched length of string s after removing all common substrings.
        :param s: Input string.
        :param common_substrs: List of common substrings.
        :return: Length of the unmatched part of the string.
        """
        for substr in common_substrs:
            s = s.replace(substr, '', 1)
        return len(s)

    def hamacher_product(self, a, b, p):
        """
        Compute the Hamacher product for the given unmatched lengths and parameter p.
        :param a: Unmatched length proportion of the first string.
        :param b: Unmatched length proportion of the second string.
        :param p: Parameter for Hamacher product.
        :return: Hamacher product score.
        """
        numerator = a * b
        denominator = p + (1 - p) * (a + b - a * b)
        return numerator / denominator if denominator != 0 else 0

    def diff(self, s1, s2, common_substrs, p=0.6):
        """
        Compute the difference between two strings based on the Hamacher product.
        :param s1: First string.
        :param s2: Second string.
        :param common_substrs: List of common substrings.
        :param p: Parameter for Hamacher product (default is 0.6).
        :return: Difference score.
        """
        uLen_s1 = self.unmatched_length(s1, common_substrs) / len(s1)
        uLen_s2 = self.unmatched_length(s2, common_substrs) / len(s2)

        return self.hamacher_product(uLen_s1, uLen_s2, p)

    def winkler(self, s1, s2):
        """
        Compute the Jaro-Winkler similarity between two strings.
        :param s1: First string.
        :param s2: Second string.
        :return: Jaro-Winkler similarity score.
        """
        return textdistance.JaroWinkler().similarity(s1, s2)

    def PMI(self, s1, s2):
        """
        Compute the Pointwise Mutual Information (PMI) between two strings.
        :param s1: First string.
        :param s2: Second string.
        :return: PMI score.
        """
        common_substrs = self.find_all_largest_common_substrings(s1, s2)
        result = self.comm(s1, s2, common_substrs) - \
            self.diff(s1, s2, common_substrs) + self.winkler(s1, s2)

        # Normalize the result to the range [0, 1]
        normalize_result = (result + 1) / (2 + 1)
        return normalize_result
