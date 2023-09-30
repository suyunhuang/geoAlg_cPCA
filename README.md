# geoAlg_cPCA
Principal component analysis (PCA) has been widely used in exploratory data analysis. Contrastive PCA (Abid et al., 2018), a generalized method of PCA, is a new tool used to capture features of a target dataset relative to a background dataset while preserving the maximum amount of information contained in the data. With high dimensional data, the original algorithm of contrastive PCA becomes impractical due to its high computational requirement of forming the contrastive covariance matrix and associated eigenvalue decomposition for extracting leading components. We propose a geometric curvilinear-search method, based on optimization in Stiefel manifold, to solve cPCA in high dimension. 

Demo example: example_mnist.m
