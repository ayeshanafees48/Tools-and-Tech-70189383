from sklearn.model_selection import cross_val_score

class CrossValidation:

    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    # -----------------------------
    # 5-Fold Cross Validation
    # -----------------------------
    def FiveFold(self):
        scores = cross_val_score(self.model, self.X, self.y, cv=5)
        return scores

    # -----------------------------
    # 10-Fold Cross Validation
    # -----------------------------
    def TenFold(self):
        scores = cross_val_score(self.model, self.X, self.y, cv=10)
        return scores

    # -----------------------------
    # Custom Fold Cross Validation
    # -----------------------------
    def CustomFold(self, folds):
        scores = cross_val_score(self.model, self.X, self.y, cv=folds)
        return scores