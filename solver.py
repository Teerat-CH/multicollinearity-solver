import pandas as pd
import networkx as nx

class Solver:
    @staticmethod
    def solve(X: pd.DataFrame, feature_importance=None, by="variance", threshold=0.9, n_select=1):
        corr_matrix = X.corr().abs()
        features = X.columns.tolist()

        print(corr_matrix)

        G = nx.Graph()
        G.add_nodes_from(features)

        correlated_features = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix.iloc[i, j] > threshold:
                    f1, f2 = features[i], features[j]
                    G.add_edge(f1, f2)
                    correlated_features.update([f1, f2])

        groups = list(nx.connected_components(G))
        print(groups)
        selected_features = []

        for group in groups:
            group = list(group)
            if len(group) <= n_select:
                selected_features.extend(group)
                continue

            if by.lower() == "variance":
                variances = X[group].var()
                best_features = variances.nlargest(n_select).index.tolist()
                selected_features.extend(best_features)
            elif by.lower() == "importance":
                if feature_importance is None:
                    raise ValueError("feature_importance must be provided when by='importance'")
                importance_sub = feature_importance.loc[group]
                best_features = importance_sub.nlargest(n_select).index.tolist()
                selected_features.extend(best_features)
            else:
                raise ValueError("Invalid 'by' argument. Use 'Variance' or 'Importance'.")

        print("Selected features:", selected_features)
        features_to_remove = list(correlated_features - set(selected_features))

        return features_to_remove
    
if __name__ == "__main__":
    data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [1, 2, 1, 2],
        'C': [4, 5, 6, 7],
        'D': [1, 1, 1, 1]
    })
    
    feature_importance = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.7, 'D': 0.2})
    
    solver = Solver()
    features_to_remove = solver.solve(data, feature_importance=feature_importance, by="importance", threshold=0.8)
    print("Features to remove:", features_to_remove)