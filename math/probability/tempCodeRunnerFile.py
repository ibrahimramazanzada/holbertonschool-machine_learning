        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        # Argument for the erf function
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Broken down Taylor series for erf to satisfy E501
        erf_series = (z - (z ** 3) / 3 + (z ** 5) / 10 -
                    (z ** 7) / 42 + (z ** 9) / 216)
        erf = (2 / (pi ** 0.5)) * erf_series

        return 0.5 * (1 + erf)