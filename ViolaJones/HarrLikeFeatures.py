import objectDet.ViolaJones.Integral as ii

    def get_score(self, int_img):
        """
        Get score for given integral image array.
        :param int_img: Integral image array, ndarray
        :return: Score for given feature, float
        """
        score = 0
        if self.type == FeatureType.TWO_VERTICAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                   self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                   self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                   (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = ii.sum_region(int_img, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                                  self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = ii.sum_region(int_img, self.top_left,
                                  (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                   (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
            third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                                  self.bottom_right)
            score = first - second + third
        # elif self.type == FeatureType.FOUR:
        #     # top left area
        #     first = ii.sum_region(int_img, self.top_left,
        #                           (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
        #     # top right area
        #     second = ii.sum_region(int_img, (int(self.top_left[0] + self.width / 2), self.top_left[1]),
        #                            (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
        #     # bottom left area
        #     third = ii.sum_region(int_img, (self.top_left[0], int(self.top_left[1] + self.height / 2)),
        #                           (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
        #     # bottom right area
        #     fourth = ii.sum_region(int_img,
        #                            (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
        #                            self.bottom_right)
        #     score = first - second - third + fourth
        return score