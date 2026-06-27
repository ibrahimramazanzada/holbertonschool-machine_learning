    def suspects(self, explanatory, n_suspects):
        '''Returns the n_suspects rows with smallest mean depth and their depths.'''
        depths = self.predict(explanatory)
        sorted_indices = np.argsort(depths)[:n_suspects]
        return explanatory[sorted_indices], depths[sorted_indices]